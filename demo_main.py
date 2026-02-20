import argparse
import time
import torch
import os
import pandas as pd
from torch_geometric.data import Dataset, DataLoader
from lib.dataset_load import load_train_test_data, load_entire_data
from lib.dataset_graph import get_individual_graph
from lib.loss import ClassWeightedFocalLoss
from lib.model_fusion import FusionMRINetSep1
from utils.statics import cal_confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score


class FusionMRIData(Dataset):
    def __init__(self, gp, clinic, ct, ca, cv, sv, sc_graph, fc_graph):
        self.gp = gp
        self.clinic = clinic
        self.ct = ct
        self.ca = ca
        self.cv = cv
        self.sv = sv
        self.sc_graph = sc_graph
        self.fc_graph = fc_graph

    def __len__(self):
        return len(self.gp)

    def __getitem__(self, idx):
        return (self.gp[idx],
                self.clinic[idx],
                self.ct[idx],
                self.ca[idx],
                self.cv[idx],
                self.sv[idx],
                self.sc_graph[idx],
                self.fc_graph[idx])


def train_fusionMRI(loader, model, criterion, optimizer, args):
    model.train()
    total_loss = 0.0
    y_pred, y_binary = [], []

    device = args.device

    for targets, clinic, ct, ca, cv, sv, sc_graph, fc_graph in loader:
        targets = targets.to(device)
        clinic = clinic.to(device)
        ct = ct.to(device)
        ca = ca.to(device)
        cv = cv.to(device)
        sv = sv.to(device)
        sc_graph = sc_graph.to(device)
        fc_graph = fc_graph.to(device)

        optimizer.zero_grad()

        outputs = model(clinic, ct, ca, cv, sv, sc_graph, fc_graph, args)  # (N, 1)
        out = outputs['output'].squeeze()  # (N, )

        binary_labels = targets  # (N, )
        loss = criterion(out, binary_labels)
        total_loss += loss.item() * targets.shape[0]

        loss.backward()
        optimizer.step()

        try:
            out1 = torch.sigmoid(out)
            y_pred.extend(out1)
            y_binary.extend(binary_labels)
        except:
            continue

    y_pred, y_binary = torch.tensor(y_pred).to(torch.float), torch.tensor(y_binary).to(torch.float)
    acc, sens, prec, f1, spec, bal_acc, g_bal_acc = cal_confusion_matrix(y_binary, y_pred)

    y_binary = y_binary.tolist()
    y_pred = y_pred.tolist()
    auc_roc = roc_auc_score(y_binary, y_pred)
    auc_pr = average_precision_score(y_binary, y_pred)

    metrics = {
        'loss': total_loss / len(loader.dataset),
        'accuracy': acc,
        'sensitivity': sens,
        'precision': prec,
        'f1_score': f1,
        'specificity': spec,
        'balanced_accuracy': bal_acc,
        'g_balanced_accuracy': g_bal_acc,
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr)
    }
    return metrics


def test_fusionMRI(loader, model, criterion, args):
    model.eval()
    total_loss = 0.0
    y_pred, y_binary = [], []

    device = args.device

    with torch.no_grad():
        for targets, clinic, ct, ca, cv, sv, sc_graph, fc_graph in loader:
            targets = targets.to(device)
            clinic = clinic.to(device)
            ct = ct.to(device)
            ca = ca.to(device)
            cv = cv.to(device)
            sv = sv.to(device)
            sc_graph = sc_graph.to(device)
            fc_graph = fc_graph.to(device)

            outputs = model(clinic, ct, ca, cv, sv, sc_graph, fc_graph, args)  # (N, 1)
            out = outputs['output'].squeeze()  # (N, )

            binary_labels = targets  # (N, )
            loss = criterion(out, binary_labels)
            total_loss += loss.item() * targets.shape[0]

            try:
                out1 = torch.sigmoid(out)
                y_pred.extend(out1)
                y_binary.extend(binary_labels)
            except:
                continue

        y_pred, y_binary = torch.tensor(y_pred).to(torch.float), torch.tensor(y_binary).to(torch.float)
        acc, sens, prec, f1, spec, bal_acc, g_bal_acc = cal_confusion_matrix(y_binary, y_pred)

        y_binary = y_binary.tolist()
        y_pred = y_pred.tolist()
        auc_roc = roc_auc_score(y_binary, y_pred)
        auc_pr = average_precision_score(y_binary, y_pred)

    metrics = {
        'loss': total_loss / len(loader.dataset),
        'accuracy': acc,
        'sensitivity': sens,
        'precision': prec,
        'f1_score': f1,
        'specificity': spec,
        'balanced_accuracy': bal_acc,
        'g_balanced_accuracy': g_bal_acc,
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr)
    }
    return metrics


def test_fusionMRISep_save(loader, model, criterion, args):
    model.eval()
    total_loss = 0.0
    y_preds = []
    y_binary_labels = []
    y_embeds = []
    y_ages = []
    y_sexs = []
    y_races = []

    device = args.device
    out_dir = args.out_dir
    save_weight = args.save_weight
    save_prob = args.save_prob
    save_name = args.save_name

    if save_weight == 1:
        df_sc = pd.DataFrame(columns=['Batch', 'Source', 'Target', 'Weight'])
        df_fc = pd.DataFrame(columns=['Batch', 'Source', 'Target', 'Weight'])
        x_ct_topk_indices = []
        x_ct_topk_weights = []
        x_ca_topk_indices = []
        x_ca_topk_weights = []
        x_cv_topk_indices = []
        x_cv_topk_weights = []
        x_sv_topk_indices = []
        x_sv_topk_weights = []

    with torch.no_grad():
        batch_idx = 0
        for targets, clinic, ct, ca, cv, sv, sc_graph, fc_graph in loader:
            targets = targets.to(device)
            clinic = clinic.to(device)
            ct = ct.to(device)
            ca = ca.to(device)
            cv = cv.to(device)
            sv = sv.to(device)
            sc_graph = sc_graph.to(device)
            fc_graph = fc_graph.to(device)

            outputs = model(clinic, ct, ca, cv, sv, sc_graph, fc_graph, args)  # (N, 1)
            out = outputs['output'].squeeze()  # (N, )
            embed = outputs['fused_embed']

            if save_prob == 1:
                if args.data_name == 'dHCP':
                    # Age, Sex, Race
                    age = clinic[:, 0]
                    sex = clinic[:, 2]
                    race = clinic[:, 3]
                else:
                    # Age, Sex, Race
                    age = clinic[:, 0]
                    sex = clinic[:, 1]
                    race = clinic[:, 2]

            if save_weight == 1:
                # T1
                ct_topk_indices = outputs['ct_topk_indices']
                ct_topk_weights = outputs['ct_topk_weights']
                ca_topk_indices = outputs['ca_topk_indices']
                ca_topk_weights = outputs['ca_topk_weights']
                cv_topk_indices = outputs['cv_topk_indices']
                cv_topk_weights = outputs['cv_topk_weights']
                sv_topk_indices = outputs['sv_topk_indices']
                sv_topk_weights = outputs['sv_topk_weights']
                x_ct_topk_indices.extend(ct_topk_indices)
                x_ct_topk_weights.extend(ct_topk_weights)
                x_ca_topk_indices.extend(ca_topk_indices)
                x_ca_topk_weights.extend(ca_topk_weights)
                x_cv_topk_indices.extend(cv_topk_indices)
                x_cv_topk_weights.extend(cv_topk_weights)
                x_sv_topk_indices.extend(sv_topk_indices)
                x_sv_topk_weights.extend(sv_topk_weights)

                # SC
                sc_attn_weights = model.model_sc.get_attention_weights()
                sc_topk_indices = (sc_attn_weights[1][0]).cpu().tolist()
                sc_topk_weights = (sc_attn_weights[1][1].mean(dim=1)).cpu().tolist()

                batch_df_sc = pd.DataFrame({
                    'Batch': [batch_idx] * len(sc_topk_weights),
                    'Source': sc_topk_indices[0],
                    'Target': sc_topk_indices[1],
                    'Weight': sc_topk_weights
                })
                df_sc = pd.concat([df_sc, batch_df_sc], ignore_index=True)

                # FC
                fc_attn_weights = model.model_fc.get_attention_weights()
                fc_topk_indices = (fc_attn_weights[1][0]).cpu().tolist()
                fc_topk_weights = (fc_attn_weights[1][1].mean(dim=1)).cpu().tolist()

                batch_df_fc = pd.DataFrame({
                    'Batch': [batch_idx] * len(fc_topk_weights),
                    'Source': fc_topk_indices[0],
                    'Target': fc_topk_indices[1],
                    'Weight': fc_topk_weights
                })
                df_fc = pd.concat([df_fc, batch_df_fc], ignore_index=True)

            binary_labels = targets  # (N, )
            loss = criterion(out, binary_labels)
            total_loss += loss.item() * targets.shape[0]
            batch_idx += 1

            try:
                out1 = torch.sigmoid(out)
                y_preds.extend(out1)
                y_binary_labels.extend(binary_labels)
                y_embeds.extend(embed)

                y_ages.extend(age)
                y_sexs.extend(sex)
                y_races.extend(race)
            except:
                continue

        y_embeds = torch.stack(y_embeds)
        y_embeds_cpu = y_embeds.cpu()
        y_embeds_npy = y_embeds_cpu.tolist()

        y_preds = torch.tensor(y_preds).to(torch.float)
        y_binary_labels = torch.tensor(y_binary_labels).to(torch.float)

        acc, sens, prec, f1, spec, bal_acc, g_bal_acc = cal_confusion_matrix(y_binary_labels, y_preds)

        y_binary_labels = y_binary_labels.tolist()
        y_preds = y_preds.tolist()
        auc_roc = roc_auc_score(y_binary_labels, y_preds)
        auc_pr = average_precision_score(y_binary_labels, y_preds)

        y_ages = torch.tensor(y_ages).to(torch.float).tolist()
        y_sexs = torch.tensor(y_sexs).to(torch.float).tolist()
        y_races = torch.tensor(y_races).to(torch.float).tolist()

        metrics = {
            'loss': total_loss / len(loader.dataset),
            'accuracy': acc,
            'sensitivity': sens,
            'precision': prec,
            'f1_score': f1,
            'specificity': spec,
            'balanced_accuracy': bal_acc,
            'g_balanced_accuracy': g_bal_acc,
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr)
        }

        results = {
            'y_feature': y_embeds_npy,
            'y_pred': y_preds,
            'y_binary': y_binary_labels,
        }

        if save_weight == 1:
            # T1
            x_ct_topk_indices = torch.stack(x_ct_topk_indices)
            x_ct_topk_indices = x_ct_topk_indices.cpu().tolist()
            x_ct_topk_weights = torch.stack(x_ct_topk_weights)
            x_ct_topk_weights = x_ct_topk_weights.cpu().tolist()

            x_ca_topk_indices = torch.stack(x_ca_topk_indices)
            x_ca_topk_indices = x_ca_topk_indices.cpu().tolist()
            x_ca_topk_weights = torch.stack(x_ca_topk_weights)
            x_ca_topk_weights = x_ca_topk_weights.cpu().tolist()

            x_cv_topk_indices = torch.stack(x_cv_topk_indices)
            x_cv_topk_indices = x_cv_topk_indices.cpu().tolist()
            x_cv_topk_weights = torch.stack(x_cv_topk_weights)
            x_cv_topk_weights = x_cv_topk_weights.cpu().tolist()

            x_sv_topk_indices = torch.stack(x_sv_topk_indices)
            x_sv_topk_indices = x_sv_topk_indices.cpu().tolist()
            x_sv_topk_weights = torch.stack(x_sv_topk_weights)
            x_sv_topk_weights = x_sv_topk_weights.cpu().tolist()

            df_ct_indices = pd.DataFrame(x_ct_topk_indices)
            df_ct_weights = pd.DataFrame(x_ct_topk_weights)

            df_ca_indices = pd.DataFrame(x_ca_topk_indices)
            df_ca_weights = pd.DataFrame(x_ca_topk_weights)

            df_cv_indices = pd.DataFrame(x_cv_topk_indices)
            df_cv_weights = pd.DataFrame(x_cv_topk_weights)

            df_sv_indices = pd.DataFrame(x_sv_topk_indices)
            df_sv_weights = pd.DataFrame(x_sv_topk_weights)

            file_name = f'{save_name}_t1_attn_weights.xlsx'
            save_dir = out_dir + file_name
            with pd.ExcelWriter(save_dir, engine='xlsxwriter') as writer:
                df_ct_indices.to_excel(writer, sheet_name='CT_TopK_Indices', index=False)
                df_ct_weights.to_excel(writer, sheet_name='CT_TopK_Weights', index=False)
                df_ca_indices.to_excel(writer, sheet_name='CA_TopK_Indices', index=False)
                df_ca_weights.to_excel(writer, sheet_name='CA_TopK_Weights', index=False)
                df_cv_indices.to_excel(writer, sheet_name='CV_TopK_Indices', index=False)
                df_cv_weights.to_excel(writer, sheet_name='CV_TopK_Weights', index=False)
                df_sv_indices.to_excel(writer, sheet_name='SV_TopK_Indices', index=False)
                df_sv_weights.to_excel(writer, sheet_name='SV_TopK_Weights', index=False)
            print(f"Saved to {save_dir}.")

            file_name = f'{save_name}_sc_attn_weights.csv'
            save_dir = out_dir + file_name
            df_sc.to_csv(save_dir, index=False)
            print(f'Save to {save_dir}.')

            file_name = f'{save_name}_fc_attn_weights.csv'
            save_dir = out_dir + file_name
            df_fc.to_csv(save_dir, index=False)
            print(f'Save to {save_dir}.')

        if save_prob == 1:
            # save preds and labels
            df = pd.DataFrame({
                'Age': y_ages,
                'Sex': y_sexs,
                'Race': y_races,
                'Label': y_binary_labels,
                'Pred': y_preds
            })
            file_name = f'{save_name}_probability.xlsx'
            save_dir = out_dir + file_name
            df.to_excel(save_dir, index=False)
            print(f'Save to {save_dir}.')

    return metrics, results


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='train or test')
    parser.add_argument('--fusion_mode', type=str, default='all', help='all or mri_only')
    parser.add_argument('--run_i', type=int, default='7', help='')
    parser.add_argument('--pretrain', type=int, default='0', help='0-no pretrain, 1-with pretrain')

    parser.add_argument('--rand_seed', type=int, default='0', help='')
    parser.add_argument('--model_type', type=int, default='1', help='1-gat, 2-gatv2')
    parser.add_argument('--abs_flag', type=int, default='1', help='0-raw, 1-abs')

    parser.add_argument('--data_name', type=str, default='ABCD', help='ABCD')
    parser.add_argument('--sub_name', type=str, default='ABCD_demo', help='ABCD_YR0')
    parser.add_argument('--device', type=str, default='cuda:3', help='cuda:0 to 3, CUDA_VISIBLE_DEVICES')

    parser.add_argument('--out_dir', type=str, default='./checkpoints/fusionMRI', help='save path')
    parser.add_argument('--save_epoch', type=int, default=5, help='save epoch')
    parser.add_argument('--save_weight', type=int, default='0', help='0=not save FC/SC/T1; 1=save FC/SC/T1')
    parser.add_argument('--save_prob', type=int, default='1', help='')
    parser.add_argument('--save_name', type=str, default='', help='save name')

    parser.add_argument('--epoch_num', type=int, default=50, help='train epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='train batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning_rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2 weight rate')

    parser.add_argument('--t1_topk_ratio', type=float, default=0.1, help='t1 topk features')
    parser.add_argument('--t1_sv_dim', type=int, default=16, help='t1_sv_dim')
    parser.add_argument('--sc_topk_ratio', type=float, default=0.1, help='SC topk nonzero ratio mask')
    parser.add_argument('--fc_topk_ratio', type=float, default=0.1, help='FC topk nonzero ratio mask')
    parser.add_argument('--conn_dim', type=int, default=268, help='conn_dim 268/64')
    parser.add_argument('--conn_hidden_dim', type=int, default=64, help=' 64/32')
    args = parser.parse_args()

    print(args.mode)
    print(args.fusion_mode)
    if args.mode == 'train':
        print(f'pretrain: ', args.pretrain)
    print(args.device)
    print(args.sub_name)

    print(f'run_i: ', args.run_i)
    print(f'rand_seed: ', args.rand_seed)
    print(f'model_type: ', args.model_type)
    print(f'abs_flag: ', args.abs_flag)

    print(f'lr: {args.lr}')
    print(f'l2: {args.l2}')
    print(f't1_topk_ratio: {args.t1_topk_ratio}')
    print(f'sc_topk_ratio: {args.sc_topk_ratio}')
    print(f'fc_topk_ratio: {args.fc_topk_ratio}')

    args.out_dir = args.out_dir + (f'_ABCD_YR0'
                                   f'_r{args.run_i}'
                                   f'_s{args.rand_seed}'
                                   f'_gs{args.model_type}'
                                   f'_a{args.abs_flag}'
                                   f'_pt{args.pretrain}'
                                   f'_lr{args.lr}'
                                   f'_t1_{args.t1_topk_ratio}'
                                   f'_sc_{args.sc_topk_ratio}'
                                   f'_fc_{args.fc_topk_ratio}/')
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'out_dir: {args.out_dir}')


    if args.data_name == 'ABCD' and args.sub_name == 'ABCD_demo':
        print(f'# ------------------------ load dataset ---------------------- #')
        entire_data = load_entire_data(args.data_name, args.sub_name)

        print(f'# ------------------------ preprocess graph dataset ---------------------- #')
        # SC
        entire_sc_graph = [get_individual_graph(sc, args.abs_flag, args.sc_topk_ratio) for sc in entire_data['sc_tensor']]
        print('entire_sc_graph: ', len(entire_sc_graph))

        # FC
        entire_fc_graph = [get_individual_graph(fc, args.abs_flag, args.fc_topk_ratio) for fc in entire_data['fc_tensor']]
        print('entire_fc_graph: ', len(entire_fc_graph))

        print(f'# ------------------------ organize loader ---------------------- #')
        entire_dataset = FusionMRIData(entire_data['label_tensor'],
                                      entire_data['clinic_tensor'],
                                      entire_data['ct_tensor'],
                                      entire_data['ca_tensor'],
                                      entire_data['cv_tensor'],
                                      entire_data['sv_tensor'],
                                      entire_sc_graph,
                                      entire_fc_graph)

        entire_loader = DataLoader(entire_dataset, batch_size=args.batch_size, shuffle=False)

        print(f'# ------------------------ define loss ---------------------- #')
        class_weight_pos = 1 - entire_data['risk_r']
        criterion = ClassWeightedFocalLoss(alpha=class_weight_pos, gamma=2)
    else:
        print(f'# ------------------------ load dataset ---------------------- #')
        train_data, test_data = load_train_test_data(args.data_name, args.sub_name, args.rand_seed)


        print(f'# ------------------------ preprocess graph dataset ---------------------- #')
        # SC
        train_sc_graph = [get_individual_graph(sc, args.abs_flag, args.sc_topk_ratio) for sc in train_data['train_sc_tensor']]
        test_sc_graph = [get_individual_graph(sc, args.abs_flag, args.sc_topk_ratio) for sc in test_data['test_sc_tensor']]
        print('train_sc_graph: ', len(train_sc_graph))
        print('test_sc_graph: ', len(test_sc_graph))

        # FC
        train_fc_graph = [get_individual_graph(fc, args.abs_flag, args.fc_topk_ratio) for fc in train_data['train_fc_tensor']]
        test_fc_graph = [get_individual_graph(fc, args.abs_flag, args.fc_topk_ratio) for fc in test_data['test_fc_tensor']]
        print('train_fc_graph: ', len(train_fc_graph))
        print('test_fc_graph: ', len(test_fc_graph))


        print(f'# ------------------------ organize loader ---------------------- #')
        train_dataset = FusionMRIData(train_data['train_label_tensor'],
                                      train_data['train_clinic_tensor'],
                                      train_data['train_ct_tensor'],
                                      train_data['train_ca_tensor'],
                                      train_data['train_cv_tensor'],
                                      train_data['train_sv_tensor'],
                                      train_sc_graph,
                                      train_fc_graph)

        test_dataset = FusionMRIData(test_data['test_label_tensor'],
                                     test_data['test_clinic_tensor'],
                                     test_data['test_ct_tensor'],
                                     test_data['test_ca_tensor'],
                                     test_data['test_cv_tensor'],
                                     test_data['test_sv_tensor'],
                                     test_sc_graph,
                                     test_fc_graph)

        if args.mode == 'train':
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        print(f'# ------------------------ define loss ---------------------- #')
        class_weight_pos = 1 - train_data['train_risk_r']
        criterion = ClassWeightedFocalLoss(alpha=class_weight_pos, gamma=2)


    print(f'# ------------------------ define model ---------------------- #')
    if args.model_type == 1:
        model = FusionMRINetSep1(t1_topk_ratio=args.t1_topk_ratio,
                                 t1_sv_dim=args.t1_sv_dim,
                                 conn_dim=args.conn_dim,
                                 conn_hidden_dim=args.conn_hidden_dim,
                                 fusion_mode=args.fusion_mode).to(args.device)

    if args.mode == 'train':
        print("Starting Training Loop...")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, mode='max', patience=5, factor=0.5, min_lr=1e-5)

        for epoch in range(args.epoch_num):
            train_metrics = train_fusionMRI(train_loader, model, criterion, optimizer, args)
            test_metrics = test_fusionMRI(test_loader, model, criterion, args)

            scheduler.step(test_metrics['balanced_accuracy'])
            # scheduler.step(test_metrics['auc_roc'])
            # if optimizer.param_groups[-1]['lr']<1.1e-5:
            #     break

            if epoch == 0:
                df_train = pd.DataFrame([train_metrics.keys(), train_metrics.values()])
                df_test = pd.DataFrame([test_metrics.keys(), test_metrics.values()])
            else:
                train_row = pd.DataFrame([train_metrics.values()])
                df_train = pd.concat([df_train, train_row], ignore_index=True)

                test_row = pd.DataFrame([test_metrics.values()])
                df_test = pd.concat([df_test, test_row], ignore_index=True)

            if epoch == 0:
                best_bal_acc = test_metrics['balanced_accuracy']
                best_epoch = epoch + 1
                save_path = args.out_dir + 'best_bal_acc.pkl'
                torch.save(model.state_dict(), save_path)

                best_g_bal_acc = test_metrics['g_balanced_accuracy']
                best_g_epoch = epoch + 1
                save_path = args.out_dir + 'best_g_bal_acc.pkl'
                torch.save(model.state_dict(), save_path)
            else:
                if test_metrics['balanced_accuracy'] > best_bal_acc:
                    best_bal_acc = test_metrics['balanced_accuracy']
                    best_epoch = epoch + 1
                    save_path = args.out_dir + 'best_bal_acc.pkl'
                    torch.save(model.state_dict(), save_path)

                if test_metrics['g_balanced_accuracy'] > best_g_bal_acc:
                    best_g_bal_acc = test_metrics['g_balanced_accuracy']
                    best_g_epoch = epoch + 1
                    save_path = args.out_dir + 'best_g_bal_acc.pkl'
                    torch.save(model.state_dict(), save_path)

            if (epoch + 1) % args.save_epoch == 0:
                save_path = args.out_dir + f'checkpoint_{epoch + 1}_epoch.pkl'
                torch.save(model.state_dict(), save_path)

            print(f'Epoch [{epoch + 1}/{args.epoch_num}]')
            print(train_metrics)
            print(test_metrics)
            print(f'best_test: {best_epoch:.0f}, {best_bal_acc:.4f}')
            print(f'best_g_test: {best_g_epoch:.0f}, {best_g_bal_acc:.4f}')

        print("End Training Loop...")

    else:
        print("Starting Testing Loop...")
        model_path = args.out_dir + 'best_g_bal_acc.pkl'
        model.load_state_dict(torch.load(model_path))
        print('Successfully load mode from ', model_path)

        if args.data_name == 'ABCD' and args.sub_name == 'ABCD_demo':
            print(f'Testing entire {args.sub_name}:')
            args.save_name = args.sub_name
            entire_metrics, entire_results = test_fusionMRISep_save(entire_loader,
                                                              model,
                                                              criterion,
                                                              args)
            print(entire_metrics)
            df_metrics = pd.DataFrame([entire_metrics.keys(), entire_metrics.values()])

            # Save metrics to excel
            excel_file = args.out_dir + f'{args.sub_name}_metrics.xlsx'
            df_metrics.to_excel(excel_file, index=False)
        else:
            print('Testing train:')
            args.save_name = 'train'
            train_metrics, train_results = test_fusionMRISep_save(train_loader,
                                                               model,
                                                               criterion,
                                                               args)
            print(train_metrics)
            df_metrics = pd.DataFrame([train_metrics.keys(), train_metrics.values()])

            print('Testing test:')
            args.save_name = 'test'
            test_metrics, test_results = test_fusionMRISep_save(test_loader,
                                                             model,
                                                             criterion,
                                                             args)
            print(test_metrics)
            metrics_row = pd.DataFrame([test_metrics.values()])
            df_metrics = pd.concat([df_metrics, metrics_row], ignore_index=True)

            excel_file = args.out_dir + 'test_metrics.xlsx'
            df_metrics.to_excel(excel_file, index=False)


    end = time.time()
    print(f'Total time: {end - start:.2f} seconds')

    if args.mode == 'train':
        excel_file = args.out_dir + 'train_logs.xlsx'
        df_train.to_excel(excel_file, index=False)

        excel_file = args.out_dir + 'test_logs.xlsx'
        df_test.to_excel(excel_file, index=False)