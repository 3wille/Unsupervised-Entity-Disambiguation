from rouge_metric import PyRouge

metrics = ['rouge-1', 'rouge-2', 'rouge-3', 'rouge-l', 'rouge-s4', 'rouge-su4']
rouge = PyRouge(rouge_n=(1, 2, 3), rouge_l=True,
                rouge_s=True, rouge_su=True, skip_gap=4)
