"""Generate the paper's numbered-column LaTeX model summary from saved fits."""

import math


def render_latex_model_summary(panels, benefit_labels):
    count = len(panels)
    if not count or any(len(panel) != len(benefit_labels) for panel in panels):
        raise ValueError('Each model panel must contain one result per benefit.')
    progressions = [[panel[i] for panel in panels] for i in range(len(benefit_labels))]
    models = [model for progression in progressions for model in progression]
    columns = len(models)
    end_row = r' \\'
    lines = [r'\begin{table}[!htbp]', r'\centering', r'\tiny',
             r'\begin{tabular}{l' + 'c' * columns + '}', r'\toprule']
    labels = [label.replace('&', r'\&').replace('Health and Wellbeing', r'Health \& Wellbeing') for label in benefit_labels]
    lines.append(' & ' + ' & '.join(r'\multicolumn{' + str(count) + r'}{c}{\textbf{' + label + '}}' for label in labels) + end_row)
    lines.append(' '.join(r'\cmidrule(lr){' + f'{2 + i * count}-{1 + (i + 1) * count}' + '}' for i in range(len(labels))))
    lines.append(' & ' + ' & '.join(f'({i + 1})' for _ in labels for i in range(count)) + end_row)
    lines.append(r'\midrule')

    def numeric(value, precision):
        text = f'{value:.{precision}f}'
        return '$-$' + text[1:] if text.startswith('-') else text

    def coefficient(term, label, precision=3):
        cells, errors = [], []
        for model in models:
            if isinstance(model, str) or term not in model.params:
                cells.append('')
                errors.append('')
                continue
            pvalue = model.pvalues[term]
            stars = '***' if pvalue < .01 else '**' if pvalue < .05 else '*' if pvalue < .1 else ''
            cells.append(numeric(model.params[term], precision) + ('$^{' + stars + '}$' if stars else ''))
            errors.append(f'({model.bse[term]:.{precision}f})')
        lines.append(label + ' & ' + ' & '.join(cells) + end_row)
        lines.append(' & ' + ' & '.join(errors) + end_row)

    coefficient('AI ROLE', r'\textbf{AI ROLE}', 2)
    groups = [
        ('Experience (ref. None Listed)', ['0 years', '1-2 years', '3-5 years', '6-10 years', '11-20 years', '21+ years']),
        ('Education (ref. No Education Listed)', ['High school or GED', 'Associate degree', "Bachelor's degree", "Master's degree", 'Ph.D. or professional degree']),
    ]
    for heading, terms in groups:
        lines.append(r'\textbf{' + heading + '} & ' + ' & '.join([''] * columns) + end_row)
        for term in terms:
            if any(not isinstance(model, str) and term in model.params for model in models):
                coefficient(term, r'\quad ' + term)
    for term, label in [('LOG_SALARY', 'log(Salary)'), ('SP500', r'S\&P 500'), ('const', 'const')]:
        if any(not isinstance(model, str) and term in model.params for model in models):
            coefficient(term, label)
    lines.extend([r'\midrule', r'\textbf{Fixed Effects} & ' + ' & '.join([''] * columns) + end_row])
    for label, present in [('Year', lambda i: True), ('Industry', lambda i: True),
                           ('State', lambda i: i >= 1), ('SOC2', lambda i: i >= 4)]:
        cells = ['Yes' if present(i) else 'No' for _ in labels for i in range(count)]
        lines.append(r'\quad ' + label + ' & ' + ' & '.join(cells) + end_row)
    lines.append(r'\midrule')
    lines.append('Observations & ' + ' & '.join(str(int(model.nobs)) if not isinstance(model, str) and math.isfinite(model.nobs) else '' for model in models) + end_row)
    lines.append(r'Pseudo R$^2$ & ' + ' & '.join(f'{model.prsquared:.3f}' if not isinstance(model, str) else '' for model in models) + end_row)
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\caption{Regression Results - All Benefits}',
                  r'\label{tab:results_all_benefits}', r'\begin{tablenotes}', r'\small',
                  r'\item Note: *p$<$0.1; **p$<$0.05; ***p$<$0.01', r'\end{tablenotes}', r'\end{table}'])
    return '\n'.join(lines) + '\n'
