# plotly plotting functions

from copy import deepcopy

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import clear_output, display
from plotly import colors
from plotly.subplots import make_subplots
from rsatoolbox.inference import evaluate as eval
from rsatoolbox.inference.bootstrap import (bootstrap_sample,
                                            bootstrap_sample_pattern,
                                            bootstrap_sample_rdm)
from rsatoolbox.util.inference_util import all_tests, get_errorbars
from rsatoolbox.util.rdm_utils import batch_to_vectors


def traces_bar_and_scatter(eval_result, models, bar_color='blue'):

    evaluations = eval_result.evaluations.squeeze()
    subject_names = [f'Subject {i+1}' for i in range(evaluations.shape[1])]
    model_names = [model.name for model in models]
    df_evaluations = pd.DataFrame(data=evaluations, index=model_names, columns=subject_names)
    means = df_evaluations.mean(axis=1)
    sem = df_evaluations.sem(axis=1)

    bars_trace = go.Bar(
        x=model_names,
        y=means,
        showlegend=False,
        marker_color=bar_color
    )

    scatter_traces = []
    for subject in subject_names:
        if subject == "Subject 1":
            showlegend = True
        scatter_traces.append(go.Scatter(
            x=df_evaluations.index,
            y=df_evaluations[subject],
            mode='markers',
            marker=dict(size=5,
                        color='white',
                        line=dict(width=1)),
            showlegend=False
        ))
    blank_trace = go.Scatter(
        x=[None],  # This ensures the trace doesn't actually plot data
        y=[None],
        mode='markers',
        marker=dict(size=5, color='white', line=dict(width=1)),
        name='Each dot represents <br> a subject'
        )
    return bars_trace, scatter_traces, blank_trace

def plot_bars_and_scatter_from_trace(bars_trace, scatter_traces, blank_trace):
    
    fig = go.Figure()
    fig.add_trace(bars_trace)
    for trace in scatter_traces:
        fig.add_trace(trace)
    fig.add_trace(blank_trace)
    fig.update_layout(
        title="",
        xaxis_title="Model",
        yaxis_title="Cosine Similarity to Data RDMs",
        legend_title="",
        width=700,
        height=500,
        template="simple_white"
    )
    return fig


# def plot_bars_and_scatter(eval_result, models):
# # Create the bar plot
#     evaluations = eval_result.evaluations.squeeze()
#     subject_names = [f'Subject {i+1}' for i in range(evaluations.shape[1])]
#     model_names = [model.name for model in models]
#     df_evaluations = pd.DataFrame(data=evaluations, index=model_names, columns=subject_names)
#     means = df_evaluations.mean(axis=1)
#     sem = df_evaluations.sem(axis=1)
    
#     fig = go.Figure()

#     # Add bars for the means
#     fig.add_trace(go.Bar(
#         x=model_names,
#         y=means,
#         showlegend=False
#     ))
        
#     for subject in ["Subject 1", "Subject 2", "Subject 3"]:
#         if subject == "Subject 1":
#             showlegend = True
#         fig.add_trace(go.Scatter(
#             x=df_evaluations.index,
#             y=df_evaluations[subject],
#             mode='markers',
#             marker=dict(size=5,
#                         color='white',
#                         line=dict(width=1)),
#             showlegend=False
#         ))

#     fig.add_trace(go.Scatter(
#         x=[None],  # This ensures the trace doesn't actually plot data
#         y=[None],
#         mode='markers',
#         marker=dict(size=5, color='white', line=dict(width=1)),
#         name='Each dot represents <br> a subject'
#     ))

#     fig.update_layout(
#         title="",
#         xaxis_title="Model",
#         yaxis_title="Cosine Similarity to Data RDMs",
#         legend_title="",
#         width=700,
#         height=500,
#         template="simple_white"
#     )
#     return fig


def convert_result_to_list_of_dicts(result):
    means = result.get_means()
    sems = result.get_sem()
    p_zero = result.test_zero()
    p_noise = result.test_noise()
    model_names = [model.name for model in result.models]
    
    results_list = []
    for i, model_name in enumerate(model_names):
        result_dict = {
            "Model": model_name,
            "Eval±SEM": f"{means[i]:.3f} ± {sems[i]:.3f}",
            "p (against 0)": "< 0.001" if p_zero[i] < 0.001 else f"{p_zero[i]:.3f}",
            "p (against NC)": "< 0.001" if p_noise[i] < 0.001 else f"{p_noise[i]:.3f}"
        }
        results_list.append(result_dict)
    
    return results_list

def print_results_table(table_trace):
    
    fig = go.Figure()
    fig.add_trace(table_trace)

    return fig

def get_trace_for_table(eval_result):

    results_list = convert_result_to_list_of_dicts(eval_result)

    table_trace = go.Table(
        header=dict(values=["Model", "Eval ± SEM", "p (against 0)", "p (against NC)"]),
        cells=dict(
            values=[
                [result["Model"] for result in results_list],  # Correctly accesses each model name
                [result["Eval±SEM"] for result in results_list],  # Correctly accesses the combined Eval and SEM value
                [result["p (against 0)"] for result in results_list],  # Accesses p-value against 0
                [result["p (against NC)"] for result in results_list]  # Accesses p-value against noise ceiling
            ],
            font=dict(size=12),  # Smaller font size for the cells
            height=27  # Smaller height for the cell rows
            )
    )
    return table_trace

def get_trace_for_noise_ceiling(noise_ceiling):
    
    noise_lower = np.nanmean(noise_ceiling[0])
    noise_upper = np.nanmean(noise_ceiling[1])
    #model_names = [model.name for model in models]

    noise_rectangle = dict(
            # Rectangle reference to the axes
            type="rect",
            xref="x domain",  # Use 'x domain' to span the whole x-axis
            yref="y",  # Use specific y-values for the height
            x0=0,  # Starting at the first x-axis value
            y0=noise_lower,  # Bottom of the rectangle
            x1=1,  # Ending at the last x-axis value (in normalized domain coordinates)
            y1=noise_upper,  # Top of the rectangle
            fillcolor="rgba(128, 128, 128, 0.4)",  # Light grey fill with some transparency
            line=dict(
                width=0,
                #color="rgba(128, 128, 128, 0.5)",
            )

        )
    return noise_rectangle

def plot_bars_and_scatter_with_table(eval_result, models, method, color='blue', table = True):
    
    if method == 'cosine':
         method_name = 'Cosine Similarity'
    elif method == 'corr':
        method_name = 'Correlation distance'
    else:
        method_name = 'Comparison method?'

    if table:
        cols = 2
        subplot_titles=["Model Evaluations", "Model Statistics"]
    else:
        cols = 1
        subplot_titles=["Model Evaluations"]

    fig = make_subplots(rows=1, cols=cols, 
                        #column_widths=[0.4, 0.6],
                        subplot_titles=subplot_titles,
                        #specs=[[{"type": "bar"}, {"type": "table"}]]
                        
                            )
    
    bars_trace, scatter_traces, blank_trace = traces_bar_and_scatter(eval_result, models, bar_color=color)

    fig.add_trace(bars_trace, row=1, col=1)
    
    for trace in scatter_traces:
        fig.add_trace(trace, row=1, col=1)

    if table:
        table_trace = get_trace_for_table(eval_result)
        fig.add_trace(table_trace, row=1, col=2)

    width = 600*cols

    fig.update_layout(
        yaxis_title=f"RDM prediction accuracy <br> (across subject mean of {method_name})",
        #legend_title="",
        width=width,
        height=600,
        template="plotly_white"
    )
    #fig.add_trace(blank_trace, row=1, col=1)
    

    return fig

def add_noise_ceiling_to_plot(fig, noise_ceiling):
        
        rectangle = get_trace_for_noise_ceiling(noise_ceiling)
        fig.add_shape(rectangle, row=1, col=1)
        return fig


def bar_bootstrap_interactive(human_rdms, models_to_compare, method):
    
    color = 'orange'
    
    button = widgets.Button(
    description="New Bootstrap Sample",
    layout=widgets.Layout(width='auto', height='auto')  # Adjust width and height as needed
    )

    #button.style.button_color = 'lightblue'  # Change the button color as you like
    button.style.font_weight = 'bold'
    button.layout.width = '300px'  # Make the button wider
    button.layout.height = '48px'  # Increase the height for a squarer appearance
    button.layout.margin = '0 0 0 0'  # Adjust margins as needed
    button.layout.border_radius = '12px'  # Rounded corners for the button

    output = widgets.Output(layout={'border': '1px solid black'})

    def generate_plot(bootstrap=False):
        if bootstrap:
                boot_rdms, idx = bootstrap_sample_rdm(human_rdms, rdm_descriptor='subject')
                result = eval.eval_fixed(models_to_compare, boot_rdms, method=method)
        else:
                result = eval.eval_fixed(models_to_compare, human_rdms, method=method)
        
        with output:
            clear_output(wait=True)  # Make sure to clear previous output first
        
            fig = plot_bars_and_scatter_with_table(result, models_to_compare, method, color)
            fig.update_layout(height=600, width=1150,
                              title=dict(text = f"Performance of Model layers for a random bootstrap sample of subjects",
                              x=0.5, y=0.95,
                              font=dict(size=20)))
            fig.show()  # Display the figure within the `with` context


    def on_button_clicked(b):
        generate_plot(bootstrap=True)

    # Now, let's create a VBox to arrange the button above the output
    vbox_layout = widgets.Layout(
        display='flex',
        flex_flow='column',
        align_items='stretch',
        width='100%',
    )


    output = widgets.Output(layout={'border': '1px solid black'})
    button.on_click(lambda b: generate_plot(bootstrap=True))  # Generate plot on button click
    vbox = widgets.VBox([button, output], layout=vbox_layout)

    # Display everything
    #display(vbox)
    display(button, output)

    generate_plot(bootstrap=False)


import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_rdm_plotly(rdms, pattern_descriptor=None, cmap='Greys',
                    rdm_descriptor=None, n_column=None, n_row=None,
                    show_colorbar=False, gridlines=None, figsize=(None, None),
                    vmin=None, vmax=None):
    # Determine the number of matrices
    mats = rdms.get_matrices()
    n_matrices = mats.shape[0]


    # Determine the number of subplots
    if n_row is None or n_column is None:
        # Calculate rows and columns to fit all matrices in a roughly square layout
        n_row = 1
        n_column = n_matrices
        
        # n_side = int(n_matrices ** 0.5)
        # n_row = n_side if n_side ** 2 >= n_matrices else n_side + 1
        # n_column = n_row if n_row * (n_row - 1) < n_matrices else n_row - 1

    subplot_size = 150
    fig_width = n_column * subplot_size
    fig_height = n_row * subplot_size
    subplot_titles = [f'{rdm_descriptor } {rdms.rdm_descriptors[rdm_descriptor][i]}' for i in range(n_matrices)] if rdm_descriptor else None
    # Create subplots
    fig = make_subplots(rows=n_row, cols=n_column, 
                        subplot_titles=subplot_titles,
                        shared_xaxes=True, shared_yaxes=True,
                        horizontal_spacing=0.02, vertical_spacing=0.1)

    # Iterate over RDMs and add them as heatmaps
    for index in range(n_matrices):
        row, col = divmod(index, n_column)
        fig.add_trace(
            go.Heatmap(z=mats[index], 
                       colorscale=cmap, 
                       showscale=show_colorbar, 
                       zmin=vmin, zmax=vmax),
            row=row+1, col=col+1
        )

    fig.update_layout(height=290, width=fig_width)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)


    #fig.show()
    return fig

def show_rdm_plotly_interactive_bootstrap_patterns(rdms, pattern_descriptor=None, cmap='Greys',
                    rdm_descriptor=None, n_column=None, n_row=None,
                    show_colorbar=False, gridlines=None, figsize=(None, None),
                    vmin=None, vmax=None):
    
    
    button = widgets.Button(
    description="New Bootstrap Sample",
    layout=widgets.Layout(width='auto', height='auto')  # Adjust width and height as needed
    )

    #button.style.button_color = 'lightblue'  # Change the button color as you like
    button.style.font_weight = 'bold'
    button.layout.width = '300px'  # Make the button wider
    button.layout.height = '48px'  # Increase the height for a squarer appearance
    button.layout.margin = '0 0 0 0'  # Adjust margins as needed
    button.layout.border_radius = '12px'  # Rounded corners for the button

    #output = widgets.Output(layout={'border': '1px solid black'})
    output = widgets.Output()

    def generate_plot(bootstrap=False):
        if bootstrap:
                im_boot_rdms, pattern_idx = bootstrap_sample_pattern(rdms, pattern_descriptor='index')
        else:
                im_boot_rdms = rdms
        
        with output:
            clear_output(wait=True)  # Make sure to clear previous output first
        
            fig = show_rdm_plotly(im_boot_rdms.subset('roi', 'FFA'), rdm_descriptor='subject')
            fig.update_layout(title=dict(text = f"Bootstrapped sample of patterns",
                                    x=0.5, y=0.95,
                                    font=dict(size=20)))
            fig.show()


    def on_button_clicked(b):
        generate_plot(bootstrap=True)

    # Now, let's create a VBox to arrange the button above the output
    vbox_layout = widgets.Layout(
        display='flex',
        flex_flow='column',
        align_items='stretch',
        width='100%',
    )

    button.on_click(lambda b: generate_plot(bootstrap=True))  # Generate plot on button click
    vbox = widgets.VBox([button, output], layout=vbox_layout)

    # Display everything
    display(vbox)
    #display(button, output)

    generate_plot(bootstrap=False)
    

# Example usage:
# Assuming fmri_rdms is your object and you've already called .get_matrices() on it
# rdms = fmri_rdms.subset('roi', 'FFA').subset('stimset', 'D1').get_matrices()
# show_rdm_plotly(rdms)


# Example usage:
# Assume 'rdms' is a list of numpy arrays, each representing an RDM matrix
# show_rdm_plotly(rdms=your_rdm_data, n_row=desired_row_count, n_column=desired_column_count)







# def plot_big_plot(results, models,
#                   bars, scatter, 
#                   noise_ceiling, table, 
#                   pairwise_model_comparison,
#                   error_bars='sem',
#                   color_scale = 'Bluered'):
    
#     perf = np.mean(results.evaluations, axis=0)
    
#     n_colors_needed = len(models)
#     # Sample n_colors_needed colors from the Plasma color scale
#     col_scale = colors.get_colorscale(color_scale)  # Retrieve the color scale
#     color_indices = np.linspace(0, 1, n_colors_needed)  # Evenly spaced indices between 0 and 1
#     sampled_colors = colors.sample_colorscale(col_scale, color_indices)  # Sample colors
     
     
#     if bars & scatter & table & pairwise_model_comparison:
#         specs=[ 
#         [{"type": "xy"}, {"type": "table"}],  # First row with two cells
#         [{"type": "bar"}, {"type": 'xy'}]     # Second row with two cells
#         ]
#         rows = 2
#         cols =2
#         subplot_titles =  ('Model Comparison', "Model Statistics", '', '')
#     elif bars & scatter & table:
#         rows = 1
#         cols = 2
#         subplot_titles = ("Model Evaluations", "Model Statistics")
#         specs=[[{"type": "bar"}, {"type": "table"}]]
#     elif bars & scatter:
#         rows = 1
#         cols = 1
#         subplot_titles = ("Model Evaluations")
#         specs=[[{"type": "bar"}]]
    
#     fig = make_subplots(rows=rows, cols=cols, 
#                     #column_widths=[0.4, 0.6],
#                     subplot_titles=subplot_titles,
#                     specs=specs)
    
#     if bars & scatter:

#         fig = plot_bars(fig, results, perf, models, sampled_colors)
#         #bars_trace, scatter_traces, blank_trace = traces_bar_and_scatter(results, models)

# #        fig.add_trace(bars_trace, row=1, col=1)

#     #     fig.update_layout(
#     #     title="",
#     #     xaxis_title="Model",
#     #     yaxis_title="RDM prediction accuracy <br> (across subject mean of cosine similarity)",
#     #     legend_title="",
#     #     width=1150,
#     #     height=600,
#     #     template="plotly_white"   
#     # )
        
#     #     for trace in scatter_traces:
#     #         fig.add_trace(trace, row=1, col=1)

#     if noise_ceiling:
#         noise_rectangle = get_trace_for_noise_ceiling(results.noise_ceiling)
#         fig.add_shape(noise_rectangle, row=1, col=1)
    
#     if table:
#         table_trace = get_trace_for_table(results)
#         fig.add_trace(table_trace, row=1, col=2)

#     if pairwise_model_comparison:
#         significant = get_significant(results)
#         fig = plot_metroplot_plotly(fig, significant, perf,models, sampled_colors)
    
#     return fig

# def get_significant(result, 
#                     alpha=0.01, 
#                     multiple_pair_testing='fdr',
#                     test_type='t-test'):
        
#         evaluations = result.evaluations
#         noise_ceiling = result.noise_ceiling
#         model_var = result.model_var
#         diff_var = result.diff_var
#         noise_ceil_var = result.noise_ceil_var
#         dof = result.dof
        
#         evaluations = evaluations[~np.isnan(evaluations[:, 0])]
#         _, n_models = evaluations.shape

#         p_pairwise, p_zero, p_noise = all_tests(
#             evaluations, noise_ceiling, test_type,
#             model_var=model_var, diff_var=diff_var,
#             noise_ceil_var=noise_ceil_var, dof=dof)

#         n_tests = int((n_models ** 2 - n_models) / 2)

#         if multiple_pair_testing.lower() == 'bonferroni' or \
#            multiple_pair_testing.lower() == 'fwer':
#             significant = p_pairwise < (alpha / n_tests)
#         elif multiple_pair_testing.lower() == 'fdr':
#             ps = batch_to_vectors(np.array([p_pairwise]))[0][0]
#             ps = np.sort(ps)
#             criterion = alpha * (np.arange(ps.shape[0]) + 1) / ps.shape[0]
#             k_ok = ps < criterion
#             if np.any(k_ok):
#                 k_max = np.max(np.where(ps < criterion)[0])
#                 crit = criterion[k_max]
#             else:
#                 crit = 0
#             significant = p_pairwise < crit
#         else:
#             if 'uncorrected' not in multiple_pair_testing.lower():
#                 raise ValueError(
#                     'plot_model_comparison: Argument ' +
#                     'multiple_pair_testing is incorrectly defined as ' +
#                     multiple_pair_testing + '.')
#             significant = p_pairwise < alpha

#         return significant

# def plot_bars(fig, result, perf, models, sampled_colors , error_bars='sem', test_type='t-test'):
    
#     model_var = result.model_var
#     dof = result.dof
#     evaluations = result.evaluations
#     evaluations = evaluations[~np.isnan(evaluations[:, 0])]


#     if error_bars:
#         limits = get_errorbars(model_var, evaluations, dof, error_bars,
#                                test_type)
#         if error_bars.lower() == 'sem':
#             limits = limits[0,:]

#     for i, (perf_val, model) in enumerate(zip(perf, models)):
#         name = model.name

#         fig.add_trace(
#             go.Bar(
#                 x=[name],  # x-axis position
#                 y=[perf_val],  # Performance value
#                 error_y=dict(type='data',
#                             array=limits, visible=True, color='black'),  # Adding error bars
#                 marker_color=sampled_colors[i],  # Cycle through colors
#                 name=name
#             ),
#             row=2, col=1  # Assuming a single subplot for simplicity
#         )

#     fig.update_layout(width=600, height=700, showlegend=False, template='plotly_white')   
#     return fig 

# def plot_metroplot_plotly(original_fig, significant, perf, models, sampled_colors):
#     # First, create a deep copy of the original figure to preserve its state
#     fig = deepcopy(original_fig)

#     n_models = len(models)
#     model_names = [m.name for m in models]
#     k = 1  # Vertical position tracker
#     marker_size = 8  # Size of the markers
#     for i, (model, color) in enumerate(zip(model_names,sampled_colors)):
#     # for i, (model, color) in enumerate(zip(model_names,colors)):

#         js = np.where(significant[i, :])[0]  # Indices of models significantly different from model i
#         j_worse = np.where(perf[i] > perf)[0]

#         worse_models = [model_names[j] for j in j_worse]  # Model names that performed worse
#         metropoints = worse_models + [model]  # Model names to plot on the y-axis
#         marker_symbols = ['circle-open' if point != model else 'circle' for point in metropoints]


#         fig.add_trace(go.Scatter(
#                 y = np.repeat(model,  len(metropoints)),
#                 #y = df_model['Model2'],
#                 x = metropoints,
#                 mode = 'lines+markers',
#                 marker = dict(
#                     color = color,
#                     symbol = marker_symbols,
#                     size = 10,
#                     line = dict(width=2, color=color)
#                 ),
#                 line=dict(width=2, color=color),
#                 showlegend = False),
#                 row = 1, col = 1,
            
#             )

#     # Update y-axis to fit the wings
#     fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=1)
#     fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=1)

#     return fig 