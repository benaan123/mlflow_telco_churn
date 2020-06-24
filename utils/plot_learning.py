import matplotlib.pyplot as plt

def plot_learning(model, eval_metric):
    """ Plots the learning curve for xgboost given validation sets.
            args:
                model: A trained XGboost model
                eval_metric: The evaluation metric in question
    """
    results = model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0'][eval_metric], label="Train")
    ax.plot(x_axis, results['validation_1'][eval_metric], label="Validation")
    ax.legend()
    plt.ylabel(eval_metric)
    plt.title(f"XGboost {eval_metric}")
    return plt