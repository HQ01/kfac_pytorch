import math
import sys
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from medpy.filter.binary import largest_connected_component

sys.path.append('..')
from utils import Metric, accuracy

def train(epoch,
          model,
          optimizer, 
          preconditioner, 
          loss_func, 
          train_sampler, 
          train_loader, 
          args):

    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss') 
    # train_accuracy = Metric('train_accuracy')
    scaler = args.grad_scaler if 'grad_scaler' in args else None

    # bar_format='{l_bar}{bar:10}{r_bar}',
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        batch_idx = range(0, len(data), args.batch_size)
        for i in batch_idx:
            data_batch = data[i:i + args.batch_size]
            target_batch = target[i:i + args.batch_size]

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(data_batch)
                    loss = loss_func(output, target_batch)
            else:
                output = model(data_batch)
                loss = loss_func(output, target_batch)
            
            loss = loss / args.batches_per_allreduce

            if args.horovod:
                loss.backward()
            else:
                if i < batch_idx[-1]:
                    with model.no_sync():
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                else:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

            with torch.no_grad():            
                train_loss.update(loss)
                # train_accuracy.update(accuracy(output, target_batch))

        if args.horovod:
            optimizer.synchronize()
            if preconditioner is not None:
                preconditioner.step()
            with optimizer.skip_synchronize():
                optimizer.step()
        else:
            if preconditioner is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                preconditioner.step()
                # usage = preconditioner.memory_usage()
                # print(usage)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
    # with tqdm(total=len(train_loader),
    #           desc='Epoch {:3d}/{:3d}'.format(epoch, args.epochs),
    #           disable=(dist.get_rank() != 0)) as t:
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         if args.cuda:
    #             data, target = data.cuda(), target.cuda()
    #         optimizer.zero_grad()

    #         batch_idx = range(0, len(data), args.batch_size)
    #         for i in batch_idx:
    #             data_batch = data[i:i + args.batch_size]
    #             target_batch = target[i:i + args.batch_size]

    #             if scaler is not None:
    #                 with torch.cuda.amp.autocast():
    #                     output = model(data_batch)
    #                     loss = loss_func(output, target_batch)
    #             else:
    #                 output = model(data_batch)
    #                 loss = loss_func(output, target_batch)
                
    #             loss = loss / args.batches_per_allreduce

    #             if args.horovod:
    #                 loss.backward()
    #             else:
    #                 if i < batch_idx[-1]:
    #                     with model.no_sync():
    #                         if scaler is not None:
    #                             scaler.scale(loss).backward()
    #                         else:
    #                             loss.backward()
    #                 else:
    #                     if scaler is not None:
    #                         scaler.scale(loss).backward()
    #                     else:
    #                         loss.backward()

    #             with torch.no_grad():            
    #                 train_loss.update(loss)
    #                 # train_accuracy.update(accuracy(output, target_batch))

    #         if args.horovod:
    #             optimizer.synchronize()
    #             if preconditioner is not None:
    #                 preconditioner.step()
    #             with optimizer.skip_synchronize():
    #                 optimizer.step()
    #         else:
    #             if preconditioner is not None:
    #                 if scaler is not None:
    #                     scaler.unscale_(optimizer)
    #                 preconditioner.step()
    #                 # usage = preconditioner.memory_usage()
    #                 # print(usage)
    #             if scaler is not None:
    #                 scaler.step(optimizer)
    #                 scaler.update()
    #             else:
    #                 optimizer.step()

            # t.set_postfix_str("loss: {:.4f}, lr: {:.4f}".format(
            #         train_loss.avg,
            #         optimizer.param_groups[0]['lr']))
            #t.update(1)
    # print("epoch: {}, train_loss: {:.4f}, lr: {:.4f}".format(epoch, train_loss.avg, optimizer.param_groups[0]['lr']))

    # if args.log_writer:
        # print('train/loss', )
        # args.log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        # args.log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        #args.log_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'],
        #                            epoch)
    #    continue


def test(epoch, 
         model, 
         loss_func, 
         val_loader, 
         args):
    model.eval()
    val_loss = Metric('val_loss')
    # val_accuracy = Metric('val_accuracy')

    validation_pred, validation_true = [], []

    # bar_format='{l_bar}{bar:10}|{postfix}',
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = loss_func(output, target)
            output_np, target_np = output.detach().cpu().numpy(), target.detach().cpu().numpy()
            validation_pred.extend(
                [output_np[s] for s in range(output_np.shape[0])]
            )
            validation_true.extend(
                [target_np[s] for s in range(target_np.shape[0])]
            )

            val_loss.single_thread_update(loss)
            # val_accuracy.update(accuracy(output, target))
            if i + 1 == len(val_loader):
                mean_dsc = np.mean(
                    dsc_per_volume(
                        validation_pred,
                        validation_true,
                        val_loader.dataset.patient_slice_index,
                    )
                )
                print("epoch {}, val_loss: {:.4f}, val_mean_dsc_value: {:.4f}".format(epoch, val_loss.avg, mean_dsc))
    return val_loss.avg, mean_dsc
    # with tqdm(total=len(val_loader),
    #           desc='             '.format(epoch, args.epochs),
    #           disable=True) as t:
    #     with torch.no_grad():
    #         for i, (data, target) in enumerate(val_loader):
    #             if args.cuda:
    #                 data, target = data.cuda(), target.cuda()
    #             output = model(data)
    #             loss = loss_func(output, target)
    #             output_np, target_np = output.detach().cpu().numpy(), target.detach().cpu().numpy()
    #             validation_pred.extend(
    #                 [output_np[s] for s in range(output_np.shape[0])]
    #             )
    #             validation_true.extend(
    #                 [target_np[s] for s in range(target_np.shape[0])]
    #             )

    #             val_loss.single_thread_update(loss)
    #             # val_accuracy.update(accuracy(output, target))

    #             t.update(1)
    #             if i + 1 == len(val_loader):
    #                 mean_dsc = np.mean(
    #                     dsc_per_volume(
    #                         validation_pred,
    #                         validation_true,
    #                         val_loader.dataset.patient_slice_index,
    #                     )
    #                 )
    #                 print("epoch {}, val_loss: {:.4f}, val_mean_dsc_value: {:.4f}".format(epoch, val_loss.avg, mean_dsc))
    #                 # t.set_postfix_str("\b\b val_loss: {:.4f}, val_mean_dsc_value: {:.2f}%".format(
    #                 #         val_loss.avg,
    #                 #         mean_dsc),
    #                 #         refresh=False)

def dsc(y_pred, y_true, lcc=True):
    y_pred = np.round(y_pred).astype(int)
    y_true = np.round(y_true).astype(int)
    if lcc and np.any(y_pred): 
        y_pred = largest_connected_component(y_pred)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_slices[p]
    return dsc_list

    # if args.log_writer:
    #    continue
        #args.log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        #args.log_writer.add_scalar('val/mean_dsc', mean_dsc, epoch)
