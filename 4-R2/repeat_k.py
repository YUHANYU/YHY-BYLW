from config import Config
config = Config()

def repeat_k(config):
    """
    推理需要，对测试的输入文件重复写k次，对测试的目标候选文件写k次
    :param infer_input:
    :param infer_target:
    :return:
    """
    orig_infer_dia = open(config.infer_input, 'r', encoding='utf-8')
    with open(config.infer_input_k, 'w', encoding='utf-8') as dia_file:
        for line in orig_infer_dia:
            for _ in range(config.k):
                dia_file.write(line.split('\n')[0])
                dia_file.write('\n')

    candidate_infer_input_con = open(config.candidate_k, 'r', encoding='utf-8')
    with open(config.infer_target_k, 'w', encoding='utf-8') as con_file:
        for line in candidate_infer_input_con:
            con_file.write(line.split('\t')[1].split('\n')[0])
            con_file.write('\n')

    # FIXME 完这两个文件需要手动删除最后一行，这个要处理！！！


repeat_k(config)  # dd50 k=25;dd200 k=60;dp50 k=30;dp200 k =70
