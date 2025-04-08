def call_seaborn(
        func_name: str
        , data
        , *args
        , save_path=None
        , show=True
        , **kwargs
        ):
    """ Wrapper method for Seaborn visualizations """
    import matplotlib
    matplotlib.use("TkAgg")

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Get the function from seaborn by name
    seaborn_func = getattr(sns, func_name)

    plot = seaborn_func(data, *args, **kwargs)

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    
    return plot
