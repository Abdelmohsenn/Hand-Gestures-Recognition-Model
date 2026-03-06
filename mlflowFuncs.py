import mlflow
import mlflow.sklearn

def startMlflowExperiment(name):
    mlflow.set_experiment(name)


def log_experiment(
    run_name,
    model,
    params,
    metrics,
    artifacts=None, 
    dataset_path=None,
    model_name="model"
):
    with mlflow.start_run(run_name=run_name):
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        if dataset_path:
            mlflow.log_artifact(dataset_path, artifact_path="dataset")
        if artifacts:
            for a in artifacts:
                mlflow.log_artifact(a)
        mlflow.sklearn.log_model(model, model_name)

def register_best_model(run_id, model_name, registered_name):
    model_uri = f"runs:/{run_id}/{model_name}"
    mlflow.register_model(model_uri, registered_name)
    
    
def get_best_run(experiment_name, metric="test_accuracy"):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(experiment.experiment_id, order_by=[f"metrics.{metric} DESC"])
    return runs.iloc[0]


def log_comparison_chart(chart_path):
    """Log the final model comparison chart as a standalone run"""
    with mlflow.start_run(run_name="Model_Comparison"):
        mlflow.log_artifact(chart_path)