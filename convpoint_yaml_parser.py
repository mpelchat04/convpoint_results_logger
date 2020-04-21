import yaml
import argparse
from pathlib import Path
import matplotlib.pyplot as mp
import numpy as np
from matplotlib.gridspec import GridSpec


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", default=f"D:/Travail/UdeS_maitrise/data/convpoint_test/results/CMM_2018_ERD_2018_aoi1")
    args = parser.parse_args()
    return args


def read_parameters(param_file):
    """Read and return parameters in .yaml file
    Args:
        param_file: Full file path of the parameters file
    Returns:
        yaml CommentedMap dict-like object
    """
    with open(param_file) as yamlfile:
        params = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return params


def open_logs(file, mode):
    with open(file=file, mode='r') as f:
        lines = f.read().splitlines()
        if mode == 1:
            last_lines = lines[-4:]
        else:
            last_lines = lines[-6:]
        return last_lines


def set_plot():
    fig = mp.figure(constrained_layout=True, figsize=(20, 20))
    gs = GridSpec(9, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, :], title='Training Loss - 4 Classes', xlim=(0, 50), ylim=(0, 0.00001))
    ax2 = fig.add_subplot(gs[2:4, :], title='Validation Loss - 4 Classes', xlim=(0, 50), ylim=(0, 0.00001))
    ax3 = fig.add_subplot(gs[4:6, :], title='Training Loss - 6 Classes', xlim=(0, 50), ylim=(0, 0.00001))
    ax4 = fig.add_subplot(gs[6:8, :], title='Validation Loss - 6 Classes', xlim=(0, 50), ylim=(0, 0.00001))
    return ax1, ax2, ax3, ax4


class Metrics():
    def __init__(self, file_path):
        self.acc_tables = {'1': {}, '2': {}}
        self.fscore_tables = {'1': {}, '2': {}}
        self.iou_tables = {'1': {}, '2': {}}
        self.markdown_file = open(file_path, 'w')

    def add_to_table(self, data, exp_name, metric, mode):
        if mode == 1:
            data_str = f'|{exp_name}|{float(data[0].split()[-1]):.3f}|{float(data[1].split()[-1]):.3f}|' \
                f'{float(data[2].split()[-1]):.3f}|{float(data[3].split()[-1]):.3f}|'
        elif mode == 2:
            data_str = f'|{exp_name}|{float(data[0].split()[-1]):.3f}|{float(data[1].split()[-1]):.3f}|' \
                f'{float(data[2].split()[-1]):.3f}|{float(data[3].split()[-1]):.3f}|{float(data[4].split()[-1]):.3f}|{float(data[5].split()[-1]):.3f}'
        else:
            raise ValueError(f'Provided mode is not implemented')
        if metric == 'iou':
            self.iou_tables[str(mode)][exp_name] = data_str
        elif metric == 'fscore':
            self.fscore_tables[str(mode)][exp_name] = data_str
        elif metric == 'acc':
            self.acc_tables[str(mode)][exp_name] = data_str
        else:
            raise Exception

    def write_to_file(self, mode):
        mode = str(mode)
        if mode == '1':
            template_table = {'title': f'|Name|Other|Building|Water|Ground',
                              'format': f'|----|----|----|----|----|'}
            self.markdown_file.write(f'# Metrics for 4 classes \n ## IOU  \n')
        else:
            template_table = {'title': f'|Name|Other|Building|Water|Ground|Low Vegetation|Medium-High Vegetation|',
                              'format': f'|----|----|----|----|----|----|----|'}
            self.markdown_file.write(f'# Metrics for 6 classes \n ## IOU  \n')

        self.markdown_file.write(template_table['title'] + '  \n' + template_table['format'] + '  \n')
        for k, v in self.iou_tables[mode].items():
            self.markdown_file.write(f'{v}  \n')

        self.markdown_file.write(f'## F1-Score  \n')
        self.markdown_file.write(template_table['title'] + '  \n' + template_table['format'] + '  \n')
        for k, v in self.fscore_tables[mode].items():
            self.markdown_file.write(f'{v}  \n')

        self.markdown_file.write(f'## Accuracy  \n')
        self.markdown_file.write(template_table['title'] + '  \n' + template_table['format'] + '  \n')
        for k, v in self.acc_tables[mode].items():
            self.markdown_file.write(f'{v}  \n')


class Configs():
    def __init__(self, file_path):
        tables = {}
        for i in ['1', '2']:
            tables[i] = {"name": f"|Name|",
                         "format": f"|----|",
                         "batchsize": f"|BatchSize|",
                         "npoints": f"|NPoints|",
                         "blocksize": f"|BlockSize|",
                         "iter": f"|Iterations|",
                         "features": f"|Features|",
                         "test_step": f"|Test Step|",
                         "nepochs": f"|Epoch|",
                         "lr": f"|LR|",
                         "model": f"|Model|",
                         "num_workers": f"|Num workers|",
                         "drop": f"|Drop|"}
        self.tables = tables
        self.markdown_file = open(file_path, 'w')
        self.data = []

    def exp_to_tables(self):
        for exp in self.data:
            data = exp['data']
            exp_name = exp['exp_name']
            mode = exp['mode']
            self.tables[str(mode)]['name'] += f"{exp_name}|"
            self.tables[str(mode)]['format'] += f"----|"
            for k, v in self.tables.items():
                if k == str(mode):
                    for key, value in v.items():
                        if key in data:
                            self.tables[k][key] += f"{data[key]} |"
                        elif key not in ['name', 'format']:
                            self.tables[k][key] += f" |"

    def add_to_exp(self, data, exp_name):
        self.data.append({'data': data, 'exp_name': exp_name, 'mode': data['mode']})

    def write_to_file(self):
        for k, v in self.tables.items():
            if k == '1':
                n_classes = 3
            else:
                n_classes = 5
            self.markdown_file.write(f"# Experiments on {n_classes} classes\n  ")
            for key, value in v.items():
                self.markdown_file.write(value + '\n')
            self.markdown_file.write('\n')

        self.markdown_file.write(f"  \n # Loss figure  \n ![image](./loss.png)")
        self.markdown_file.close()


def main():
    args = arg_parse()
    root_path = Path(args.rootdir)
    project_name = root_path.parts[-1]
    configs = Configs(file_path=root_path / f'{project_name}_config.md')
    val_metrics = Metrics(file_path=root_path / f'{project_name}_val_metrics.md')
    trn_metrics = Metrics(file_path=root_path / f'{project_name}_trn_metrics.md')
    tst_metrics = Metrics(file_path=root_path / f'{project_name}_tst_metrics.md')
    ax1, ax2, ax3, ax4 = set_plot()
    for file_path in root_path.glob('**/*.yaml'):
        yaml_dict = read_parameters(file_path)
        exp_name = file_path.parent.parts[-1]
        configs.add_to_exp(yaml_dict, exp_name)
        mode = yaml_dict['mode']

        exp_path = file_path.parent
        for metrics_file in exp_path.glob('**/*.log*'):
            split_name = metrics_file.stem.split('_')
            if split_name[1] == 'classwise':
                log_file = open_logs(metrics_file, mode)
                metric_name = split_name[-1]
                if split_name[2] == 'val':
                    val_metrics.add_to_table(log_file, exp_name, metric_name, mode)
                elif split_name[2] == 'trn':
                    trn_metrics.add_to_table(log_file, exp_name, metric_name, mode)
                elif split_name[2] == 'tst':
                    tst_metrics.add_to_table(log_file, exp_name, metric_name, mode)
            else:
                if split_name[-1] == 'loss':
                    with open(file=metrics_file, mode='r') as f:
                        lines = f.read().splitlines()
                    if len(lines) > 1:
                        new_list = [float(x.split()[1]) for x in lines]
                        np_log_data = np.asarray(new_list)
                        if split_name[1] == 'trn' and mode == 1:
                            ax1.plot(np.arange(0, len(np_log_data), 1), np_log_data, label=exp_name)
                        elif split_name[1] == 'val' and mode == 1:
                            ax2.plot(np.arange(0, len(np_log_data), 1), np_log_data, label=exp_name)
                        if split_name[1] == 'trn' and mode == 2:
                            ax3.plot(np.arange(0, len(np_log_data), 1), np_log_data, label=exp_name)
                        elif split_name[1] == 'val' and mode == 2:
                            ax4.plot(np.arange(0, len(np_log_data), 1), np_log_data, label=exp_name)

    mp.legend(bbox_to_anchor=(0, -0.1), loc='upper left', ncol=2)
    # mp.show()
    mp.savefig(root_path / f'loss.png')

    val_metrics.write_to_file(mode=1)
    val_metrics.write_to_file(mode=2)
    trn_metrics.write_to_file(mode=1)
    trn_metrics.write_to_file(mode=2)
    tst_metrics.write_to_file(mode=1)
    tst_metrics.write_to_file(mode=2)
    configs.exp_to_tables()
    configs.write_to_file()


if __name__ == '__main__':
    main()
