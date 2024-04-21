with plt.xkcd():
    fig, axs = plt.subplots(1,2,figsize = (15,5), sharey = True)
    axs[0].hist(frame.flatten(), bins = 100);
    axs[1].hist(frame_HT.flatten(), bins = 100);

    #utils
    [ax.set_yscale('log') for ax in axs]
    [remove_edges(ax) for ax in axs]
    [add_labels(ax, ylabel = 'Count', xlabel = 'Value') for ax in axs]
    [ax.set_title(title) for title, ax in zip(['Before Thresholding', 'After Thresholding'], axs)]