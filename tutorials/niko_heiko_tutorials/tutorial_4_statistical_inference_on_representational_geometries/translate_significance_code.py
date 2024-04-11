
'''
@author: vb
mar 2024'''

import warnings
from copy import deepcopy

# import ipdb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.colors
import plotly.graph_objects as go
from matplotlib import cm, patches, transforms
from matplotlib.path import Path
from networkx.algorithms.clique import find_cliques as maximal_cliques
from plotly.express import colors
from plotly.subplots import make_subplots
from rsatoolbox.util.inference_util import all_tests, get_errorbars
from rsatoolbox.util.rdm_utils import batch_to_vectors
from scipy.spatial.distance import squareform


def plot_model_comparison_trans(result, sort=False, colors=None,
                          alpha=0.01, test_pair_comparisons=True,
                          multiple_pair_testing='fdr',
                          test_above_0=True,
                          test_below_noise_ceil=True,
                          error_bars='sem',
                          test_type='t-test'):
    

    # Prepare and sort data
    evaluations = result.evaluations
    models = result.models
    noise_ceiling = result.noise_ceiling
    method = result.method
    model_var = result.model_var
    diff_var = result.diff_var
    noise_ceil_var = result.noise_ceil_var
    dof = result.dof

    while len(evaluations.shape) > 2:
        evaluations = np.nanmean(evaluations, axis=-1)

    evaluations = evaluations[~np.isnan(evaluations[:, 0])]
    n_bootstraps, n_models = evaluations.shape
    perf = np.mean(evaluations, axis=0)

    noise_ceiling = np.array(noise_ceiling)
    sort = 'unsorted'
    # run tests
    if any([test_pair_comparisons,
            test_above_0, test_below_noise_ceil]):
        p_pairwise, p_zero, p_noise = all_tests(
            evaluations, noise_ceiling, test_type,
            model_var=model_var, diff_var=diff_var,
            noise_ceil_var=noise_ceil_var, dof=dof)
        
    if error_bars:
        limits = get_errorbars(model_var, evaluations, dof, error_bars,
                               test_type)
        if error_bars.lower() == 'sem':
            limits = limits[0,:]
        
    #return limits, perf
        
    fig = make_subplots(rows=2, cols=1, 
                        row_heights=[0.3, 0.7],
                        vertical_spacing=0.05,
                        subplot_titles=("Model Evaluations", ''),
                        shared_xaxes=True,
                        )
    
    
    # antique_colors = plotly.colors.qualitative.Antique  # Get the Antique color palette
    # n_colors = len(antique_colors)  # Number of colors in the palette

    n_colors_needed = len(models)
    # Sample n_colors_needed colors from the Plasma color scale
    plasma_scale = plotly.colors.get_colorscale('Bluered')  # Retrieve the color scale
    color_indices = np.linspace(0, 1, n_colors_needed)  # Evenly spaced indices between 0 and 1
    sampled_colors = plotly.colors.sample_colorscale(plasma_scale, color_indices)  # Sample colors
    
    for i, (perf_val, model) in enumerate(zip(perf, models)):
        name = model.name
        #bar_color = antique_colors[i % n_colors]

        fig.add_trace(
            go.Bar(
                x=[name],  # x-axis position
                y=[perf_val],  # Performance value
                error_y=dict(type='data',
                            array=limits, visible=True, color='black'),  # Adding error bars
                marker_color=sampled_colors[i],  # Cycle through colors
                name=name
            ),
            row=2, col=1  # Assuming a single subplot for simplicity
        )


    fig.update_layout(width=600, height=700, showlegend=False, template='plotly_white')    
    # return fig


    model_significant = p_zero < alpha / n_models
    significant_indices = [i for i, significant in enumerate(model_significant) if significant]
    symbols = {'dewdrops': 'circle', 'icicles': 'diamond-tall'}

    fig.add_trace(
        go.Scatter(
            x=[models[i].name for i in significant_indices],  # X positions of significant models
            y=[0.0005] * len(significant_indices),  # Y positions (at 0 for visualization)
            mode='markers',
            marker=dict(symbol=symbols['dewdrops'],  # Example using 'triangle-up'
                        size=9, 
                        color='white'),  # Example using 'triangle-up'
            showlegend=False
        ),
        row=2, col=1
    )

    # Plot noise ceiling
    if noise_ceiling is not None:

        noise_lower = np.nanmean(noise_ceiling[0])
        noise_upper = np.nanmean(noise_ceiling[1])
        model_names = [model.name for model in models]

        fig.add_shape(
                # Rectangle reference to the axes
                type="rect",
                xref="x domain",  # Use 'x domain' to span the whole x-axis
                yref="y",  # Use specific y-values for the height
                x0=0,  # Starting at the first x-axis value
                y0=noise_lower,  # Bottom of the rectangle
                x1=1,  # Ending at the last x-axis value (in normalized domain coordinates)
                y1=noise_upper,  # Top of the rectangle
                fillcolor="rgba(128, 128, 128, 0.5)",  # Light grey fill with some transparency
                line=dict(
                    color='gray',
                ),
                opacity=0.5,
                layer="below",  # Ensure the shape is below the data points
                row=2, col=1  # Specify the subplot where the shape should be added

            )

    test_below_noise_ceil = 'dewdrops'  # Example, can be True/'dewdrops'/'icicles'
    model_below_lower_bound = p_noise < (alpha / n_models)

    significant_indices_below = [i for i, below in enumerate(model_below_lower_bound) if below]

    # Choose the symbol based on the test_below_noise_ceil
    if test_below_noise_ceil is True or test_below_noise_ceil.lower() == 'dewdrops':
        symbol = 'circle-open'  # Use open circle as a proxy for dewdrops
    elif test_below_noise_ceil.lower() == 'icicles':
        symbol = 'diamond-open'  # Use open diamond as a proxy for icicles
    else:
        raise ValueError('Argument test_below_noise_ceil is incorrectly defined as ' + test_below_noise_ceil)
    
    symbol = 'triangle-down'
#    y_position_below = noise_lower + 0.0005  # Adjust based on your visualization needs

    #y_positions_below = [perf[i] for i in significant_indices_below]  # Extracting perf values for significant models
    y_positions_below = [noise_lower-0.005] * len(significant_indices_below)  # Adjust based on your visualization needs
    fig.add_trace(
        go.Scatter(
            x=[models[i].name for i in significant_indices_below],  # X positions of significant models
            y= y_positions_below, #* len(significant_indices_below),  # Y positions slightly above noise_lower
            mode='markers',
            marker=dict(symbol=symbol, size=7, color='gray'),  # Customizing marker appearance
            showlegend=False
        ),
        row=2, col=1
    )

    #return fig

    # Pairwise model comparisons
    if test_pair_comparisons:
        if test_type == 'bootstrap':
            model_comp_descr = 'Model comparisons: two-tailed bootstrap, '
        elif test_type == 't-test':
            model_comp_descr = 'Model comparisons: two-tailed t-test, '
        elif test_type == 'ranksum':
            model_comp_descr = 'Model comparisons: two-tailed Wilcoxon-test, '
        n_tests = int((n_models ** 2 - n_models) / 2)
        if multiple_pair_testing is None:
            multiple_pair_testing = 'uncorrected'
        if multiple_pair_testing.lower() == 'bonferroni' or \
           multiple_pair_testing.lower() == 'fwer':
            significant = p_pairwise < (alpha / n_tests)
        elif multiple_pair_testing.lower() == 'fdr':
            ps = batch_to_vectors(np.array([p_pairwise]))[0][0]
            ps = np.sort(ps)
            criterion = alpha * (np.arange(ps.shape[0]) + 1) / ps.shape[0]
            k_ok = ps < criterion
            if np.any(k_ok):
                k_max = np.max(np.where(ps < criterion)[0])
                crit = criterion[k_max]
            else:
                crit = 0
            significant = p_pairwise < crit
        else:
            if 'uncorrected' not in multiple_pair_testing.lower():
                raise ValueError(
                    'plot_model_comparison: Argument ' +
                    'multiple_pair_testing is incorrectly defined as ' +
                    multiple_pair_testing + '.')
            significant = p_pairwise < alpha
        model_comp_descr = _get_model_comp_descr(
            test_type, n_models, multiple_pair_testing, alpha,
            n_bootstraps, result.cv_method, error_bars,
            test_above_0, test_below_noise_ceil)
        

        # new_fig_nili = plot_nili_bars_plotly(fig, significant, models, version=1)
        # new_fig_gol = plot_golan_wings_plotly(fig, significant, perf, models)

        new_fig_metro = plot_metroplot_plotly(fig, significant, perf, models, sampled_colors)

        return new_fig_metro
        

    # MATPLOT LIB CODE STILL TO INSPECT
    #     fig.suptitle(model_comp_descr, fontsize=fs2/2)
    #     axbar.set_xlim(ax.get_xlim())
    #     digits = [d for d in list(test_pair_comparisons) if d.isdigit()]
    #     if len(digits) > 0:
    #         v = int(digits[0])
    #     else:
    #         v = None
    #     if 'nili' in test_pair_comparisons.lower():
    #         if v:
    #             plot_nili_bars(axbar, significant, version=v)
    #         else:
    #             plot_nili_bars(axbar, significant)
    #     elif 'golan' in test_pair_comparisons.lower():
    #         if v:
    #             plot_golan_wings(axbar, significant, perf, sort, colors,
    #                              version=v)
    #         else:
    #             plot_golan_wings(axbar, significant, perf, sort, colors)
    #     elif 'arrows' in test_pair_comparisons.lower():
    #         plot_arrows(axbar, significant)
    #     elif 'cliques' in test_pair_comparisons.lower():
    #         plot_cliques(axbar, significant)

    # # Floating axes
    # if method == 'neg_riem_dist':
    #     ytoptick = noise_upper + 0.1
    #     ymin = np.min(perf)
    # else:
    #     ytoptick = np.floor(min(1, noise_upper) * 10) / 10
    #     ymin = 0
    # ax.set_yticks(np.arange(ymin, ytoptick + 1e-6, step=0.1))
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.set_xticks(np.arange(n_models))
    # ax.spines['left'].set_bounds(ymin, ytoptick)
    # ax.spines['bottom'].set_bounds(0, n_models - 1)
    # ax.yaxis.set_ticks_position('left')
    # ax.xaxis.set_ticks_position('bottom')
    # plt.rc('ytick', labelsize=fs2)

    # # Axis labels
    # y_label_string = _get_y_label(method)
    # ylabel_fig_x, ysublabel_fig_x = 0.07, 0.095
    # trans = transforms.blended_transform_factory(fig.transFigure,
    #                                              ax.get_yaxis_transform())
    # ax.text(ylabel_fig_x, (ymin + ytoptick) / 2, 'RDM prediction accuracy',
    #         horizontalalignment='center', verticalalignment='center',
    #         rotation='vertical', fontsize=fs, fontweight='bold',
    #         transform=trans)
    # ax.text(ysublabel_fig_x, (ymin+ytoptick)/2,
    #         y_label_string,
    #         horizontalalignment='center', verticalalignment='center',
    #         rotation='vertical', fontsize=fs2, fontweight='normal',
    #         transform=trans)

    # if models is not None:
    #     ax.set_xticklabels([m.name for m in models], fontsize=fs2,
    #                        rotation=45)
    # return fig, ax, axbar


def plot_golan_wings_plotly(original_fig, significant, perf, models):
    # First, create a deep copy of the original figure to preserve its state
    fig = deepcopy(original_fig)

    n_models = len(models)
    model_names = [m.name for m in models]
    # Use the Plotly qualitative color palette
    colors = plotly.colors.qualitative.Plotly
   
    k = 1  # Vertical position tracker
    marker_size = 8  # Size of the markers
    for i in range(n_models):

        js = np.where(significant[i, :])[0]  # Indices of models significantly different from model i
        if len(js) > 0:
            for j in js:
                # Ensure cycling through the color palette
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(x=[model_names[i], model_names[j]], 
                                            y=[k, k],
                                        mode='lines', 
                                        line=dict(color=color, width=2)
                                        ),
                                        row=1, col=1)
                fig.add_trace(go.Scatter(x=[model_names[i]], y=[k],
                                        mode='markers', 
                                        marker=dict(symbol='circle', color=color, size=10,
                                                    line=dict(color=color, width=2))
                                        ),
                                        row=1, col=1)
             
                if perf[i] > perf[j]: 
                    # Draw downward feather
                    fig.add_trace(go.Scatter(x=[model_names[j]], 
                                            y=[k],
                                            mode='markers',
                                            marker=dict(symbol='triangle-right', color=color, size=marker_size,
                                                        line=dict(color=color, width=2))
                                            ),
                                            row=1, col=1)
                elif perf[i] < perf[j]:
                    # Draw upward feather
                    fig.add_trace(go.Scatter(x=[model_names[i], model_names[j]], 
                                             y=[k, k],
                                            mode='lines', 
                                            line=dict(color=color, width=2)
                                            ),
                                            row=1, col=1)
                    fig.add_trace(go.Scatter(x=[model_names[j]], y=[k],
                                            mode='markers', 
                                            marker=dict(symbol='triangle-left', color=color, size=marker_size,
                                                        line=dict(color=color, width=2))
                                            ),
                                            row=1, col=1)
            k += 1  # Increment vertical position after each model's wings are drawn

    # Update y-axis to fit the wings
    fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=1)

    return fig 


def plot_metroplot_plotly(original_fig, significant, perf, models, sampled_colors):
    # First, create a deep copy of the original figure to preserve its state
    fig = deepcopy(original_fig)

    # n_colors_needed = len(models)
    # # Sample n_colors_needed colors from the Plasma color scale
    # plasma_scale = plotly.colors.get_colorscale('Bluered')  # Retrieve the color scale
    # color_indices = np.linspace(0, 1, n_colors_needed)  # Evenly spaced indices between 0 and 1
    # sampled_colors = plotly.colors.sample_colorscale(plasma_scale, color_indices)  # Sample colors

    n_models = len(models)
    model_names = [m.name for m in models]
    # Use the Plotly qualitative color palette
    colors = plotly.colors.qualitative.Antique
   
    k = 1  # Vertical position tracker
    marker_size = 8  # Size of the markers
    for i, (model, color) in enumerate(zip(model_names,sampled_colors)):
    # for i, (model, color) in enumerate(zip(model_names,colors)):

        js = np.where(significant[i, :])[0]  # Indices of models significantly different from model i
        j_worse = np.where(perf[i] > perf)[0]

        worse_models = [model_names[j] for j in j_worse]  # Model names that performed worse
        metropoints = worse_models + [model]  # Model names to plot on the y-axis
        #marker_symbols = ['circle-open' if point != model else 'circle' for point in metropoints]
        marker_colors = ['white' if point != model else color for point in metropoints]  # Fill color for markers



        fig.add_trace(go.Scatter(
                y = np.repeat(model,  len(metropoints)),
                #y = df_model['Model2'],
                x = metropoints,
                mode = 'lines+markers',
                marker = dict(
                    color = marker_colors,
                    symbol = 'circle',
                    size = 10,
                    line = dict(width=2, color=color)
                ),
                line=dict(width=2, color=color),
                showlegend = False),
                row = 1, col = 1,
            
            )

    # Update y-axis to fit the wings
    fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=1)

    return fig 




def plot_nili_bars_plotly(original_fig, significant, models, version=1):

    fig = deepcopy(original_fig)

    k = 1  # Vertical position tracker
    ns_col = 'rgba(128, 128, 128, 0.5)'  # Non-significant comparison color
    w = 0.2  # Width for nonsignificant comparison tweaks
    model_names = [m.name for m in models]
    
    for i in range(significant.shape[0]):
        drawn1 = False
        for j in range(i + 1, significant.shape[0]):
            if version == 1 and significant[i, j]:
                # Draw a line for significant differences
                fig.add_shape(type="line",
                              x0=i, y0=k, x1=j, y1=k,
                              line=dict(color="black", width=2),
                              xref="x1", yref="y1",
                              row=1, col=1)
                k += 1
                drawn1 = True
            elif version == 2 and not significant[i, j]:
                # Draw a line for non-significant differences
                fig.add_shape(type="line",
                              x0=i, y0=k, x1=j, y1=k,
                              line=dict(color=ns_col, width=2),
                              xref="x1", yref="y1",
                              row=1, col=1)
                # Additional visual tweaks for non-significant comparisons
                fig.add_annotation(x=(i+j)/2, y=k, text="n.s.",
                                   showarrow=False,
                                   font=dict(size=8, color=ns_col),
                                   xref="x1", yref="y1",
                                   row=1, col=1)
                k += 1
                drawn1 = True
                
        if drawn1:
            k += 1  # Increase vertical position after each row of comparisons

    fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=1)

    fig.update_layout(height=700)  # Adjust as necessary
    return fig


def _get_model_comp_descr(test_type, n_models, multiple_pair_testing, alpha,
                          n_bootstraps, cv_method, error_bars,
                          test_above_0, test_below_noise_ceil):
    """constructs the statistics description from the parts

    Args:
        test_type : String
        n_models : integer
        multiple_pair_testing : String
        alpha : float
        n_bootstraps : integer
        cv_method : String
        error_bars : String
        test_above_0 : Bool
        test_below_noise_ceil : Bool

    Returns:
        model

    """
    if test_type == 'bootstrap':
        model_comp_descr = 'Model comparisons: two-tailed bootstrap, '
    elif test_type == 't-test':
        model_comp_descr = 'Model comparisons: two-tailed t-test, '
    elif test_type == 'ranksum':
        model_comp_descr = 'Model comparisons: two-tailed Wilcoxon-test, '
    n_tests = int((n_models ** 2 - n_models) / 2)
    if multiple_pair_testing is None:
        multiple_pair_testing = 'uncorrected'
    if multiple_pair_testing.lower() == 'bonferroni' or \
       multiple_pair_testing.lower() == 'fwer':
        model_comp_descr = (model_comp_descr
                            + 'p < {:<.5g}'.format(alpha)
                            + ', Bonferroni-corrected for '
                            + str(n_tests)
                            + ' model-pair comparisons')
    elif multiple_pair_testing.lower() == 'fdr':
        model_comp_descr = (model_comp_descr +
                            'FDR q < {:<.5g}'.format(alpha) +
                            ' (' + str(n_tests) +
                            ' model-pair comparisons)')
    else:
        if 'uncorrected' not in multiple_pair_testing.lower():
            raise ValueError(
                'plot_model_comparison: Argument ' +
                'multiple_pair_testing is incorrectly defined as ' +
                multiple_pair_testing + '.')
        model_comp_descr = (model_comp_descr +
                            'p < {:<.5g}'.format(alpha) +
                            ', uncorrected (' + str(n_tests) +
                            ' model-pair comparisons)')
    if cv_method in ['bootstrap_rdm', 'bootstrap_pattern',
                     'bootstrap_crossval']:
        model_comp_descr = model_comp_descr + \
            '\nInference by bootstrap resampling ' + \
            '({:<,.0f}'.format(n_bootstraps) + ' bootstrap samples) of '
    if cv_method == 'bootstrap_rdm':
        model_comp_descr = model_comp_descr + 'subjects. '
    elif cv_method == 'bootstrap_pattern':
        model_comp_descr = model_comp_descr + 'experimental conditions. '
    elif cv_method in ['bootstrap', 'bootstrap_crossval']:
        model_comp_descr = model_comp_descr + \
            'subjects and experimental conditions. '
    if error_bars[0:2].lower() == 'ci':
        model_comp_descr = model_comp_descr + 'Error bars indicate the'
        if len(error_bars) == 2:
            CI_percent = 95.0
        else:
            CI_percent = float(error_bars[2:])
        model_comp_descr = (model_comp_descr + ' ' +
                            str(CI_percent) + '% confidence interval.')
    elif error_bars.lower() == 'sem':
        model_comp_descr = (
            model_comp_descr +
            'Error bars indicate the standard error of the mean.')
    elif error_bars.lower() == 'sem':
        model_comp_descr = (model_comp_descr +
                            'Dots represent the individual model evaluations.')
    if test_above_0 or test_below_noise_ceil:
        model_comp_descr = (
            model_comp_descr +
            '\nOne-sided comparisons of each model performance ')
    if test_above_0:
        model_comp_descr = model_comp_descr + 'against 0 '
    if test_above_0 and test_below_noise_ceil:
        model_comp_descr = model_comp_descr + 'and '
    if test_below_noise_ceil:
        model_comp_descr = (
            model_comp_descr +
            'against the lower-bound estimate of the noise ceiling ')
    if test_above_0 or test_below_noise_ceil:
        model_comp_descr = (model_comp_descr +
                            'are Bonferroni-corrected for ' +
                            str(n_models) + ' models.')
    return model_comp_descr