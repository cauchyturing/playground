from datetime import datetime, timezone
import json
import os
from typing import NamedTuple, List, Optional, Dict

# KFP
from kfp.v2 import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import component

# GCP
from google_cloud_pipeline_components.aiplatform import TabularDatasetCreateOp

# NOTE: CustomContainerTrainingJobRunOp will be depricated in v2
# Reference: https://github.com/kubeflow/pipelines/commit/2a75e44b82b2f59f64b5044218895e132b9da97d
# Consider using Qi's custom version: https://github.aetna.com/dso/vertex-pipelines/blob/cf0694c47a65614eb290f7cfc1c33096a5fa3f0e/data/biglake_pattern/biglake/vertex_ai_training.py#L111
from google_cloud_pipeline_components.aiplatform import CustomContainerTrainingJobRunOp
from google_cloud_pipeline_components.aiplatform import ModelBatchPredictOp


# Training components
@component(
    packages_to_install=["google-cloud-bigquery"],  # TODO: fix version and make dynamic
    base_image="python:3.7.12",
)
def training_data_preparator(
    project: str,
    location: str,
    train_data_sql: str,
    train_data_gcs_uri: str,
    biglake_table_path: str,
    biglake_data_format: str,
    biglake_connection_name: str,
    labels: dict,
):
    """Create a Big Lake External table by exporting the query results of the SQL in train_data_sql to train_data_gcs_uri with the biglake_data_format using the biglake_connection_name, which has to have been created with read permissions on the bucket used in train_data_gcs_uri."""
    from google.cloud import bigquery
    from google.cloud.bigquery.table import _EmptyRowIterator

    prefix = f"""\
CREATE TEMP TABLE TEMP_TABLE 
AS
"""

    postfix = f"""\
EXPORT DATA 
OPTIONS (
    uri='{train_data_gcs_uri}',
    format = `{biglake_data_format}`,
    overwrite=true
    )
AS 
    SELECT * FROM TEMP_TABLE
"""
    final_sql = f"{prefix} (\n{train_data_sql}\n);\n{postfix}"
    print(f"Running export query:\n{final_sql}")
    client = bigquery.Client(project=project)
    job_config = bigquery.QueryJobConfig(labels=labels)
    export_query_job = client.query(final_sql, job_config)
    export_result = export_query_job.result()  # waits for job to complete
    if isinstance(
        export_result, _EmptyRowIterator
    ):  # Export was successful # Is this the best pattern?
        print(f"Exported training data to {train_data_gcs_uri}")
        create_biglake_table_query = f"""\
CREATE OR REPLACE EXTERNAL TABLE `{biglake_table_path}`
WITH CONNECTION `{biglake_connection_name}`
OPTIONS (
    uris = ['{train_data_gcs_uri}'],
    format = "{biglake_data_format}"
    
)
"""
        print(
            f"Running Big Lake External Table creation query:\n{create_biglake_table_query}"
        )
        biglake_query_job = client.query(create_biglake_table_query, job_config)
        biglake_result = biglake_query_job.result()
        if isinstance(biglake_result, _EmptyRowIterator):
            print(f"Created Big Lake External Table: {biglake_table_path}")


@component(
    packages_to_install=[
        "google-api-core==1.34.0",
        "google-api-python-client==1.12.11",
        "google-auth==2.16.0",
        "google-auth-httplib2==0.1.0",
        "google-auth-oauthlib==0.8.0",
        "google-cloud-aiplatform==1.21.0",
        "google-cloud-bigquery==3.4.2",
        "google-cloud-core==2.3.2",
        "google-cloud-datastore==1.15.5",
        "google-cloud-monitoring==2.14.1",
        "google-cloud-notebooks==1.4.4",
        "google-cloud-pipeline-components==1.0.27",
        "google-cloud-resource-manager==1.8.1",
        "google-cloud-storage==2.7.0",
        "google-crc32c==1.5.0",
        "google-resumable-media==2.4.1",
        "googleapis-common-protos==1.58.0",
    ],
    base_image="python:3.7.12",
)
def model_endpoint_monitor_creator(
    endpoint_resource_name: str,
    display_name: str,
    train_dataset_uri: str,
    target_column_name: str,
    feature_names: list,
    stats_anomalies_base_directory: str,
    project: str,
    location: str,
    encryption_spec_key_name: str,
    labels: dict = {},
    user_emails: list = [],
    default_skew_thresholds: float = 0.0,
    skew_thresholds: dict = {},
    attrib_skew_thresholds: dict = {},
    sample_rate: float = 1.0,
    monitor_interval: int = 24,
) -> NamedTuple("monitor", [("name", str)]):
    """the pipleine complonent that add model monitor to endpoint
    #TODO: update docs.
    Args:
        endpoint (str): endpoint resource name. Format: 'projects/{project}/locations/{location}/endpoints/{endpoint}'
            Example: 'projects/490430656274/locations/us-east4/endpoints/7754688384936706048'
        display_name (str): the display name of this monitor job
            Example: 'jiaq_sprint5_sklearn_cls_endpoint_monitor'
        train_dataset_uri (str): the gcs path of the training data
            Example: "project.dataset.table"
        target_column_name (str): the column name for predicting
            Example: 'y'
        feature_names (list): the column names for features, in a list
            Example: ['os','is_mobile','country','pageviews','customer_id','ts']
        stats_anomalies_base_directory (str, optional): saves anomalies to defined gcs directory.
        emails (list): the email address want to be notified, should be a list of string
            Example: ['jiaq@aetna.com']
        project (str): project name
            Example: 'anbc-pdev'
        location (str): project location
            Example: 'us-east4'
        skew_threshold (float, optional): The threshold here is against feature distribution distance between the training and prediction feature. Defaults to 0.1.
        attrib_skew_threshold (float, optional): Feature attributions indicate how much each feature in your model contributed to the predictions for each given instance. Defaults to 0.1.
        sample_rate (float, optional): Sets the sampling rate for model monitoring logs. If set to 1, all logs are processed. Defaults to 0.5.
        monitor_interval (float, optional): Sets the model monitoring job scheduling interval in hours. This defines how often the monitoring jobs are triggered. Defaults to 1, which is 1 hour.
    """  # https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.ModelDeploymentMonitoringJob#google_cloud_aiplatform_ModelDeploymentMonitoringJob
    # https://github.com/googleapis/python-aiplatform/tree/main/google/cloud/aiplatform/model_monitoring

    from google.cloud import aiplatform
    from google.cloud.aiplatform import model_monitoring

    # Initialize aiplatform
    aiplatform.init(
        project=project,
        location=location,
        encryption_spec_key_name=encryption_spec_key_name,
    )
    # Configurations
    skew_config = model_monitoring.SkewDetectionConfig(
        data_source=train_dataset_uri,
        target_field=target_column_name,
    )
    if skew_thresholds:
        setattr(skew_config, "skew_thresholds", skew_thresholds)
    if default_skew_thresholds > 0:
        setattr(skew_config, "skew_thresholds", default_skew_thresholds)
    if attrib_skew_thresholds:
        setattr(skew_config, "attrib_skew_thresholds", attrib_skew_thresholds)
    # TODO: explanation_config = model_monitoring.ExplanationConfig()
    objective_config = model_monitoring.ObjectiveConfig(skew_config)
    # Create sampling configuration
    random_sample_config = model_monitoring.RandomSampleConfig()
    if sample_rate:
        setattr(random_sample_config, "sample_rate", sample_rate)
    # Create schedule configuration
    schedule_config = model_monitoring.ScheduleConfig(monitor_interval=monitor_interval)
    # Create alerting configuration.
    if user_emails:
        email_alert_config = model_monitoring.EmailAlertConfig(
            user_emails=user_emails, enable_logging=True
        )
    # Get the endpoint uri
    temp_uri = endpoint_resource_name
    print(f"temp_uri: {temp_uri}")
    endpoint_uri = temp_uri[temp_uri.find("projects/") :]
    # endpoint_uri = temp_uri[temp_uri.find('endpoints/')+10:]
    print(f"endpoint_uri: {endpoint_uri}")
    print(f"display_name: {display_name}")
    print(f"random_sample_config: {random_sample_config.__dict__}")
    print(f"schedule_config: {schedule_config.__dict__}")
    print(f"email_alert_config: {email_alert_config.__dict__}")
    print(f"skew_config: {skew_config.__dict__}")
    print(f"objective_config: {objective_config.__dict__}")
    print(f"stats_anomalies_base_directory: {stats_anomalies_base_directory}")

    # endpoint = aiplatform.Endpoint(endpoint_name=endpoint_uri, project=project, location=location)
    # print(endpoint)
    # Create the monitoring job.
    job = aiplatform.ModelDeploymentMonitoringJob.create(
        display_name=display_name,
        logging_sampling_strategy=random_sample_config,
        schedule_config=schedule_config,
        alert_config=email_alert_config,
        objective_configs=objective_config,
        # TODO: This supposedly enable storing anomalies json result to GCS but currently not working. Google support case # : 44468679
        stats_anomalies_base_directory=stats_anomalies_base_directory,
        project=project,
        location=location,
        endpoint=endpoint_uri,
        labels=labels,
        encryption_spec_key_name=encryption_spec_key_name,
    )

    # monitor_link.uri = "https://us-east4-aiplatform.googleapis.com/v1/"+job.gca_resource.name
    monitor_name = job.gca_resource.name
    return (monitor_name,)


@component(
    # TODO: install the following packages in custom container. Installing packages during runtime create errors that seem to be related to security scan
    # packages_to_install=[
    #     "parquet",
    #     "fastparquet",
    #     "fsspec",
    # ], --> leads to thirftpy error
    base_image="us-central1-docker.pkg.dev/anbc-pdev/mleng-platform-docker-pdev/bajajv-sklearn-cpu.0-23"
)
def model_evaluator(
    eval_metrics: List[str],
    aip_test_data_uri: str,
    aip_model_dir: str,
) -> NamedTuple("Outputs", [("metrics", dict)]):
    """Compute evaluation metrics by loading model in memory

    Args:
        eval_metrics: List of evaluation metrics to compute
            Example: ["accuracy", "recall"]
        aip_test_data_uri: GCS path to test data
            Example: "gs://bucket/dir/test.parquet"
        aip_model_dir: GCS path to model directory
            Example: "gs://bucket/dir/model"

    Returns:
        metrics: Key value pair where key is metrics and value is metric value
            Example: {"accuracy": 0.8, "recall": 0.2}
    """
    from typing import Callable, Tuple
    import os

    import pandas as pd
    import numpy as np
    import google.cloud.storage as gs

    # TODO: decide the format of the following env vars
    os.environ["AIP_STORAGE_URI"] = (
        aip_model_dir + "/"
    )  # NOTE: should be like: 'gs://<bucket>/<path>/model/' as of now
    os.environ["AIP_TEST_DATA_URI"] = aip_test_data_uri + "/"
    # TODO: use values in user_config.json for the following
    os.environ["feature_names"] = "os,is_mobile,country,pageviews"

    # NOTE: Borrowing from HK's custom prediction container
    # TODO: Create a module for this class so it can be shared across the package
    class serving_model:
        def __init__(self):
            """constructor, download, load/initialize and test model"""
            import os
            import google.cloud.storage as gs

            if os.environ.get("AIP_STORAGE_URI") is not None:
                AIP_STORAGE_URI = os.environ["AIP_STORAGE_URI"]
            else:
                AIP_STORAGE_URI = "MANUAL AIP_STORAGE_URI"
            print("AIP_STORAGE_URI(AIP_MODEL_DIR from training) : ", AIP_STORAGE_URI)
            AIP_STORAGE_URI = AIP_STORAGE_URI.replace("gs://", "")
            AIP_STORAGE_URI = AIP_STORAGE_URI.rstrip("/")  # remove last slash string

            storage_client = gs.Client()

            # look through model directory and find model file name which might be either pickle or joblib
            self.model = self.load_joblib_pickle(storage_client, AIP_STORAGE_URI)

        def load_joblib_pickle(self, storage_client, AIP_STORAGE_URI):
            """returns joblib or pickle loaded model
            Args:
                AIP_STORAGE_URI (dict): AIP_STORAGE_URI, same directory as AIP_MODEL_DIR from training
            Returns:
                model (obj): model file loaded as an object
            """
            model_file_name = self.find_joblib_pickle_file_name(
                storage_client, AIP_STORAGE_URI
            )
            self.download_model_file(storage_client, AIP_STORAGE_URI, model_file_name)
            model = self.load_model(model_file_name)
            return model

        def find_joblib_pickle_file_name(
            self, storage_client, AIP_STORAGE_URI: str
        ) -> str:
            """Look into model storage folder and find either pickle, joblib ext file
            Args:
                storage_client (obj): google.cloud.storage, gs.Client()
                AIP_STORAGE_URI (str): AIP_STORAGE_URI, same directory as AIP_MODEL_DIR from training
            Returns:
                filename (str): model file name
            """
            filename = ""
            first_slash = AIP_STORAGE_URI.find("/")
            bucket_name = AIP_STORAGE_URI[:first_slash]
            blobs = storage_client.list_blobs(bucket_name)
            print(f"Check for model file in blob URI '{AIP_STORAGE_URI}' : ")

            for blob in blobs:
                # print("blob name: ", blob.name, end=", ")
                filename = blob.name[blob.name.rfind("/") + 1 :]  # filename set here
                # print("file name: ", filename)
                # print("checking file name...")
                possible_model_file_names = ["model", "joblib", "pickle", "pkl"]
                for x in possible_model_file_names:
                    if x in filename:
                        print(f"model file '{filename}' found, stop search")
                        return filename

            raise RuntimeError("No model file found")

        def download_model_file(
            self, storage_client, AIP_STORAGE_URI: str, model_file_name: str
        ):
            """Download model from GCS
            Args:
                storage_client (obj): google.cloud.storage, gs.Client()
                AIP_STORAGE_URI (str): AIP_STORAGE_URI, same directory as AIP_MODEL_DIR from training
                model_file_name (str): model file name (ex. joblib, pickle, pkl)
            Returns:
                filename (str): model file name
            """
            import os

            first_slash = AIP_STORAGE_URI.find("/")
            bucket_name = AIP_STORAGE_URI[:first_slash]
            # must be like: root_dir/.../model.joblib (without bucket name)
            blob_name = os.path.join(
                AIP_STORAGE_URI[first_slash + 1 :], model_file_name
            )
            bucket = storage_client.get_bucket(bucket_name)

            print(f"AIP_STORAGE_URI = {AIP_STORAGE_URI}")
            print(f"Bucket name = {bucket_name}")
            print(f"Blob name = {blob_name}")
            print(f"model_file_name = {model_file_name}")
            blob = bucket.get_blob(blob_name)
            blob.download_to_filename(model_file_name)
            print("Model downloaded")

        def load_model(self, model_file_name: str):
            """load downloaded model
            Args:
                model_file_name (str): model file name (ex. joblib, pickle, pkl)
            Returns:
                model (obj): model object
            """
            import joblib
            import pickle

            model = None
            print("Attept to load model file : ")

            try:
                model = pickle.load(model_file_name)
                print("loaded pickled model")
            except:
                print("Attempt load the model file using pickle but didn't work")

            try:
                model = joblib.load(model_file_name)
                print("loaded joblib model")
            except:
                print("Attempt load the model file using joblib but didn't work")

            print("Model loaded(just printing model object): ", model)
            return model

        def predict(self, features: dict) -> dict:
            """returns predicted result
            Args:
                features (dict): test input data list
            Returns:
                predict result in list
            """
            return self.model.predict(features).tolist()

    # TODO: Create a module for this function so it can be reused in OOS evaluation pipeline
    def get_metric_fn(metric: str) -> Callable:
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score

        if metric == "accuracy":
            return accuracy_score
        elif metric == "precision":
            return precision_score
        elif metric == "recall":
            return recall_score
        elif metric == "f1":
            return f1_score
        else:
            raise RuntimeError(f"Unknown metric name: {metric}")

    # TODO: use the following instead of load_jsonl once parquet data is pushed from task.py
    #     def load_parquet(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #         """Util function to load parquet file and return a X, y pair

    #         Args:
    #             file_path (str): GCS compatible file path. E.g., gs://BUCKET/BLOB/FILENAME.EXT
    #         """
    #         test_data = pd.read_parquet(file_path)
    #         X = test_data.loc[:, ~test_data.columns.isin(["y"])]
    #         y = test_data["y"]
    #         return X, y

    def load_jsonl(blob):
        """Util function to load jsonl data"""
        import json
        import pandas as pd

        Xs = []
        ys = []
        for json_str in blob.download_as_string().decode("utf-8").split("\n")[:-1]:
            # print(json_str)
            row = json.loads(json_str)
            Xs.append(row["X"])
            ys.append(row["y"])
        return pd.DataFrame(
            Xs, columns=os.environ["feature_names"].split(",")
        ), pd.DataFrame(ys)

    # Load model based on AIP_* env vars
    print("Loading model...")
    mymodel = serving_model()
    # Get a list of test files in AIP_TEST_DATA_DIR
    print("Finding test files...")
    storage_client = gs.Client()
    AIP_TEST_DATA_URI = os.environ["AIP_TEST_DATA_URI"].replace("gs://", "")
    first_slash = AIP_TEST_DATA_URI.find("/")
    bucket_name = AIP_TEST_DATA_URI[:first_slash]
    blob_prefix = AIP_TEST_DATA_URI[first_slash:].lstrip("/")
    bucket = storage_client.bucket(bucket_name)
    print("Log", blob_prefix + "test_")
    blobs = [x for x in bucket.list_blobs(prefix=blob_prefix + "test_")]
    print(f"Test files: {blobs}")
    # Iterate through files to get a list of ground truths and predictions
    print("Generating predictions...")
    ground_truths, predictions = [], []
    for blob in blobs:
        # Create GCS path
        file_path = "gs://" + blob.bucket.name + "/" + blob.name
        print(f"Generating predictions using data in {file_path}")
        # Load blob
        # X_test, y_test = load_parquet(blob)
        X_test, y_test = load_jsonl(blob)
        # Get predictions
        y_pred = mymodel.predict(X_test)
        # Append ground truths and predictions
        ground_truths.extend([int(x) for x in y_test.values.reshape(-1).tolist()])
        predictions.extend([int(x) for x in y_pred])
    # Compute metrics
    print("Computing metrics...")
    metric_result = {}
    for metric in eval_metrics:
        metric_fn = get_metric_fn(metric)
        metric_val = metric_fn(ground_truths, predictions)
        # Replace NaN with -1 since it causes error in train_metrics_writer
        if np.isnan(metric_val):
            metric_result[metric] = -1
        else:
            metric_result[metric] = round(metric_val, 4)
    print("Done!")
    return (metric_result,)


@component(
    packages_to_install=[
        "google-cloud-aiplatform",
    ],
    base_image="python:3.7.12",
)
def deploy_model_or_not(
    model_resource_name: str,
    model_version_id: str,
    metrics_result: dict,
    target_metric: str,
    threshold: float,
) -> NamedTuple("Outputs",[("decision", str),]):
    """Decide whether to deploy model or not

    Args:
        metrics_result: Key value pair where key is metrics and value is metric value
            Example: {"accuracy": 0.8, "recall": 0.2}
        target_metric: Target metric to check
            Example: "accuracy"
        threshold: If the target metric value is below this value, not deploy model
            Example: "gs://bucket/dir/test.parquet"

    Returns:
        decision: Whether to deploy model or not. JSON boolean ("true" or "false")
    """
    from google.cloud import aiplatform
    
    metric_observation = metrics_result[target_metric]
    if metric_observation >= threshold:
        decision = "true"
        # Set the model as default model
        model_registry = aiplatform.models.ModelRegistry(model_resource_name)
        model_registry.add_version_aliases(
            new_aliases=["default"],
            version=model_version_id,
        )
    else:
        decision = "false"
    return (decision,)


@component(
    packages_to_install=[
        "google-cloud-aiplatform",
        "google-cloud-bigquery",
    ],
    base_image="python:3.7.12",
)
def train_metrics_result_writer(
    metrics_result: dict,
    model_id: str,
    model_name: str,
    model_version_id: str,
    project: str,
    bq_train_metric_storage_uri: str,
) -> None:
    """Write train metrics to train storage

    Args:
        metrics_result: Key value pair where key is metrics and value is metric value
            Example: {"accuracy": 0.8, "recall": 0.2}
        model_id: ID of model in Model Registry
            Example: "1234"
        model_name: Name of model in Model Registry
            Example: "sklearn-binary-clf"
        model_version_id: Current model version ID
            Example: "1"
        project: GCP project
            Example: "anbc-dev"
        bq_train_metric_storage_uri: BQ URI to train storage
            Example: "project.dataset.table"
    """
    import warnings
    import json

    from google.cloud import bigquery
    from datetime import datetime, timezone

    # Replace NaN with -1; otherwise, BQ rejects it
    metrics_result = {k: -1 if v == "NaN" else v for k, v in metrics_result.items()}
    # Need to pass project; otherwise, it gets 403 Permission Error
    client = bigquery.Client(project=project)
    print("Writing to metric storage...")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows_to_insert = [
        {
            "model_id": model_id,
            "model_version_id": model_version_id,
            "model_name": model_name,
            "timestamp": timestamp,
            "metrics": json.dumps(metrics_result),
        }
    ]
    print(rows_to_insert)
    errors = client.insert_rows_json(bq_train_metric_storage_uri, rows_to_insert)
    if len(errors) == 0:
        print("New rows have been added")
    else:
        raise RuntimeError("Encountered errors while inserting rows: {}".format(errors))
    print("Done!")


@component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image="python:3.7.12",
)
def model_uploader(
    artifact_uri: str,
    display_name: str,
    custom_predict_image_uri: str,
    feature_names: List[str],
    labels: dict,
    version_aliases: List[str],
    project: str,
    location: str,
    encryption_spec_key_name: str,
    version_description: Optional[str] = None,
    explanation_method: Optional[str] = "shapley",
) -> NamedTuple(
    "Ouptuts",
    [
        ("model_id", str),
        ("model_dir", str),
        ("model_version_id", str),
        ("model_resource_name", str),
    ],
):
    """Upload model to Model Registry. Creates a new version if model exists; otheriwise,
    it creates a new model in the registry

    Args:
        artifact_uri:
            The path to the directory containing the Model artifact and
            any of its supporting files. Leave blank for custom container prediction
        display_name:
            The display name of the Model. The name can be up to 128
            characters long and can be consist of any UTF-8 characters
        custom_predict_image_uri:
            The URI of the Model serving container. This parameter is required
            if the parameter `local_model` is not specified
        feature_names:
            List of feature column names for the target model
        labels:
            Dict of labels for cost tracking
        version_aliases:
            User provided version aliases so that a model version
            can be referenced via alias instead of auto-generated version ID.
            A default version alias will be created for the first version of the model.

            The format is [a-z][a-zA-Z0-9-]{0,126}[a-z0-9]
        project:
            Project to upload this model to. Overrides project set in
            aiplatform.init
        location:
            Location to upload this model to. Overrides location set in
            aiplatform.init
        version_description:
            The description of the model version being uploaded

    Returns:
        Instantiated representation of the uploaded model resource

    Raises:
        ValueError: If explanation_metadata is specified while explanation_parameters
            is not.
            Also if model directory does not contain a supported model file.
            If `local_model` is specified but `serving_container_spec.image_uri`
            in the `local_model` is None.
            If `local_model` is not specified and `serving_container_image_uri`
            is None.
    """
    import logging

    from google.cloud import aiplatform

    def get_model_resource_name_or_does_not_exist(
        display_name: str,
        project: str,
        location: str,
    ) -> str:
        """Return the resource name of model or empty string if model is not in Model Registry

        Args:
            display_name: Display name in Model Registry
            project:
                Project to retrieve list from. If not set, project
                set in aiplatform.init will be used.
            location:
                Location to retrieve list from. If not set, location
                set in aiplatform.init will be used.

        Returns:
            True if model exists; otherwise, False

        Notes:
          * 'aiplatform.models.ModelRegistry' failed for this use case as it raises _InactiveRpcError if there is no model.
            And _InactiveRpcError somehow cannot be captured by try-except block.
            On the other hand, 'aiplatform.Model.list' returns empty list if no model exists
          * The 'resource_name' of the model consists of 'projects/<project>/locations/<location>/models/<model_id>'.
            In other words, it does not include 'version_id'.
        """
        models = aiplatform.Model.list(
            filter=f'display_name="{display_name}"',
            project=project,
            location=location,
        )
        if len(models) > 0:
            return models[0].resource_name
        else:
            return ""

    # Check whether model is in Model Registry
    print("Checking model in model registry...")
    model_resource_name = get_model_resource_name_or_does_not_exist(
        display_name=display_name,
        project=project,
        location=location,
    )
    # If str is empty, no model exists
    # 'parent_model' does not require version id
    if len(model_resource_name) > 0:
        print("Model exists. Uploading a new version..")
        parent_model = model_resource_name
    else:
        print("No existing model. Creating a new one..")
        parent_model = None

    # prepare for explanation arguments
    if explanation_method == "shapley":
        parameters = {"sampled_shapley_attribution": {"path_count": 10}}
    elif explanation_method == "ig":
        raise NotImplementedError(
            "ig method for model explanation has not been implemented yet"
        )
    elif explanation_method == "xrai":
        raise NotImplementedError(
            "xrai method for model explanation has not been implemented yet"
        )
    else:
        logging.warning(
            "Unrecognized explanation method received. Use shapley instead."
        )
        parameters = {"sampled_shapley_attribution": {"path_count": 10}}

    parameters = aiplatform.explain.ExplanationParameters(parameters)
    input_metadata = aiplatform.explain.ExplanationMetadata.InputMetadata()
    output_metadata = aiplatform.explain.ExplanationMetadata.OutputMetadata()
    metadata = aiplatform.explain.ExplanationMetadata(
        inputs={"features": input_metadata}, outputs={"prediction": output_metadata}
    )

    # Upload model to Model Registry
    model = aiplatform.Model.upload(
        artifact_uri=artifact_uri,
        display_name=display_name,
        is_default_version=False, # It's only default when deployed
        parent_model=parent_model,
        version_aliases=version_aliases,
        version_description=version_description,
        serving_container_image_uri=custom_predict_image_uri,
        # TODO: retrieve the routes from conduit.constants
        serving_container_health_route="/health",
        serving_container_predict_route="/predict",
        serving_container_ports=[8080],
        serving_container_environment_variables={"feature_names": feature_names},
        labels=labels,
        project=project,
        location=location,
        explanation_metadata=metadata,
        explanation_parameters=parameters,
        encryption_spec_key_name=encryption_spec_key_name,
    )
    model.wait()
    print("Done!")
    # NOTE: return artifact_uri because it's dynamically generated in runtime
    return (model.resource_name.split("/")[-1], artifact_uri, model.version_id, model.resource_name)


# TODO: need to confirm cost associated with model with 0% traffic
# TODO: undeploy 0% traffic models in the Endpoint
@component(
    packages_to_install=["google-cloud-aiplatform"],
    base_image="python:3.7.12",
)
def model_deployer(
    model_id: str,
    endpoint_display_name: str,
    project: str,
    location: str,
    labels: dict,
    min_replica_count: int,
    machine_type: str,
    enable_request_response_logging: bool,
    request_response_logging_bq_destination_table: str,
    encryption_spec_key_name: str,
) -> NamedTuple(
    "Outputs",
    [
        ("endpoint_resource_name", str),
        ("model_id", str),
        ("model_version_id", str),
    ],
):
    """Deploy the 'default' model associated with 'model_id' to the Endpoint

    Args:
        model_id: Model ID in Model registry
        endpoint_display_name: Dispaly name of the Endpoint
        project:
            Project to retrieve endpoint from. If not set, project
            set in aiplatform.init will be used
        location:
            Location to retrieve endpoint from. If not set, location
            set in aiplatform.init will be used
        labels:
            Dict of labels for cost tracking
        machine_type:
            The type of machine. Not specifying machine type will
            result in model to be deployed with automatic resources
        min_replica_count:
            The minimum number of machine replicas this deployed
            model will be always deployed on. If traffic against it increases,
            it may dynamically be deployed onto more replicas, and as traffic
            decreases, some of these extra replicas may be freed
        enable_request_response_logging:
            Prediction request is logged to request_respose_logging table.
            When this parameter is used,
            request_response_logging_bq_destination_table needs to be defined together.
            The request_response_logging table is created at the
            request_response_logging_bq_destination_table BQ URI Path.
            This table is used for drift/skew endpoint monitoring.
            The request_response_logging_bq_destination_table table is shared among
            deployments in a BQ dataset bucket.
        request_response_logging_bq_destination_table
            BQ dataset path for request_respose_logging table.
            Define path up to dataset or the endpoint deployment goes into forever deploy.
            Example: "bq://{project_id}.{dataset_name}"
    """
    from google.cloud import aiplatform

    # NOTE: assuming endpoint display name is unique across project
    # check endpoint exists
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        project=project,
        location=location,
    )
    # create endpoint if not exist
    if len(endpoints) == 0:
        print(f"Endpoint does not exist. Creating one..")
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            project=project,
            location=location,
            enable_request_response_logging=enable_request_response_logging,
            request_response_logging_bq_destination_table=request_response_logging_bq_destination_table,
            labels=labels,
            encryption_spec_key_name=encryption_spec_key_name,
        )
        endpoint.wait()
    else:
        print("Found endpoint. Continue..")
        endpoint = endpoints[0]
    # get model obj
    models = aiplatform.Model.list(
        filter=f'model="{model_id}"', project=project, location=location
    )
    if len(models) == 0:
        raise RuntimeError(f"Failed to find {model_id}")
    else:
        print(f"Deploying {models[0].resource_name}@{models[0].version_id}")
    model = models[0]
    # deploy model to endpoint
    endpoint.deploy(
        model=model,
        traffic_percentage=100,
        min_replica_count=min_replica_count,
        machine_type=machine_type,
    )
    next_version = str(model.version_id)
    print("Done!")
    return (endpoint.resource_name, model_id, next_version)


@component(
    packages_to_install=["pandas", "pyarrow", "google-cloud-aiplatform"],
    base_image="python:3.7",
)
def prediction_generator(
    project: str,
    location: str,
    job_display_name: str,
    model: str,
    bigquery_source_input_uri: str,
    bigquery_destination_output_uri: str,
    machine_type: str,
    encryption_spec_key_name: str,
    excluded_fields: list = [],
    labels: dict = {},
) -> NamedTuple("Outputs", [("batch_job_id", str), ("model_id", str),]):
    """Executes batch prediction job and stores result to specified BQ table.

    Args:
        project:
            Project to retrieve endpoint from. If not set, project
            set in aiplatform.init will be used
        location:
            Location to retrieve endpoint from. If not set, location
            set in aiplatform.init will be used
        job_display_name:
            Job display name for batch prediction job node
        model:
            String model name for batch prediction to use
        bigquery_source_input_uri:
            Input BQ table for prediction
        bigquery_destination_output_uri:
            BQ table for batch prediction result
        machine_type:
            The type of machine. Not specifying machine type will
            result in model to be deployed with automatic resources
        encryption_spec_key_name: KMS key for encryption
            Example: projects/{project-id}/locations/{locations}/keyRings/{locations}-nprod-hsm/cryptoKeys/{env}
        labels:
            Dict of labels for cost tracking

    """
    import time

    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1.types import (
        BatchPredictionJob,
        EncryptionSpec,
        BatchDedicatedResources,
        BigQuerySource,
        BigQueryDestination,
        MachineSpec,
    )
    from google.cloud.aiplatform_v1.services.job_service import JobServiceClient

    # get model resource
    aiplatform.init(project=project, location=location)
    target_model = aiplatform.Model.list(filter=f'display_name="{model}"')[0]
    instances_format = instance_type = predictions_format = "bigquery"
    batch_prediction_job = BatchPredictionJob(
        display_name=job_display_name,
        model=target_model.resource_name,
        labels=labels,
        encryption_spec=EncryptionSpec(kms_key_name=encryption_spec_key_name),
        input_config=BatchPredictionJob.InputConfig(
            instances_format=instances_format,
            bigquery_source=BigQuerySource(input_uri=bigquery_source_input_uri),
        ),
        instance_config=BatchPredictionJob.InstanceConfig(
            instance_type=instance_type,
            excluded_fields=excluded_fields,
        ),
        output_config=BatchPredictionJob.OutputConfig(
            predictions_format=predictions_format,
            bigquery_destination=BigQueryDestination(
                output_uri=bigquery_destination_output_uri
            ),
        ),
        dedicated_resources=BatchDedicatedResources(
            machine_spec=MachineSpec(machine_type=machine_type),
            starting_replica_count=1,
            max_replica_count=1,
        ),
    )
    print(f"Writing batch prediction results to {bigquery_destination_output_uri}")
    client = JobServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )
    out = client.create_batch_prediction_job(
        parent=f"projects/{project}/locations/{location}",
        batch_prediction_job=batch_prediction_job,
    )
    batch_job_id = out.name.split("/")[-1]
    # Wait till the job finish
    name = client.batch_prediction_job_path(
        project=project, location=location, batch_prediction_job=batch_job_id
    )
    response = client.get_batch_prediction_job(name=name)
    print("Job state: ", str(response.state).split(".")[-1])
    while str(response.state) == "JobState.JOB_STATE_PENDING":
        response = client.get_batch_prediction_job(name=name)
        time.sleep(1)
    print("Job state: ", str(response.state).split(".")[-1])
    while str(response.state) == "JobState.JOB_STATE_RUNNING":
        response = client.get_batch_prediction_job(name=name)
        time.sleep(1)
    print("Job state: ", str(response.state).split(".")[-1])
    print("Batch prediction job id: ", batch_job_id)
    return (batch_job_id, target_model.resource_name.split("/")[-1])


@component(
    packages_to_install=["pandas", "pyarrow", "google-cloud-aiplatform"],
    base_image="python:3.7",
)
def batch_prediction_generator_monitor(
    project: str,
    location: str,
    job_display_name: str,
    model: str,
    bigquery_source_input_uri: str,
    bigquery_destination_output_uri: str,
    train_data_uri: str,
    target_column_name: str,
    monitor_output_uri: str,
    machine_type: str,
    encryption_spec_key_name: str,
    excluded_fields: list = [],
    default_skew_threshold: float = 0.0,
    skew_thresholds: dict = {},
    attribution_score_skew_thresholds: dict = {},
    drift_thresholds: dict = {},
    user_emails: list = [],
    labels: dict = {},
) -> NamedTuple(
    "Outputs",
    [
        ("batch_job_id", str),
        ("stats_gcs_folder", str),
        ("skew_gcs_folder", str),
        ("monitor_explanation_dest_uri", str),
    ],
):
    """Executes batch prediction job with drift monitoring and stores result to specified BQ table and GCS.

    Args:
        project:
            Project to retrieve endpoint from. If not set, project
            set in aiplatform.init will be used
        location:
            Location to retrieve endpoint from. If not set, location
            set in aiplatform.init will be used
        job_display_name:
            Job display name for batch prediction job node
        model:
            String model name for batch prediction to use
        bigquery_source_input_uri:
            Input BQ table for prediction
        bigquery_destination_output_uri:
            BQ table for batch prediction result
        train_data_uri:
            Used training data for model training
        target_column_name:
            Training data target column name
        monitor_output_uri:
            GCS location uri for storing drift/skew monitoring result
        machine_type:
            The type of machine. Not specifying machine type will
            result in model to be deployed with automatic resources
        encryption_spec_key_name: KMS key for encryption
            Example: projects/{project-id}/locations/{locations}/keyRings/{locations}-nprod-hsm/cryptoKeys/{env}
        skew_threshold (float, optional): The threshold here is against feature distribution distance between the training and prediction feature. Defaults to 0.1.
        attrib_skew_threshold (float, optional): Feature attributions indicate how much each feature in your model contributed to the predictions for each given instance. Defaults to 0.1.
        drift_thresholds (float, optional): The threshold here is against feature distribution distance between the prediction feature. Defaults to 0.1.
        user_emails (list): the email address want to be notified, should be a list of string
            Example: ['hokyeong.ra@cvshealth.com']
        labels:
            Dict of labels for cost tracking

    """
    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1beta1.types import (
        BatchPredictionJob,
        EncryptionSpec,
        BatchDedicatedResources,
        BigQuerySource,
        BigQueryDestination,
        GcsDestination,
        MachineSpec,
        ModelMonitoringAlertConfig,
        ModelMonitoringConfig,
        ModelMonitoringObjectiveConfig,
        ThresholdConfig,
    )
    from google.cloud.aiplatform_v1beta1.services.job_service import JobServiceClient
    import time
    # if any of drift/skew monitoring isn't configured, kill custom component
    # TODO: due to this component used for explanation, below will be deleted
    # if (
    #     (len(skew_thresholds) == 0)
    #     and (len(attribution_score_skew_thresholds) == 0)
    #     and (len(drift_thresholds) == 0)
    #     and (default_skew_threshold == 0.0)
    # ):
    #     return ("", "", "")
    instances_format = instance_type = predictions_format = "bigquery"
    # get model resource
    aiplatform.init(project=project, location=location)
    model_resource_name = aiplatform.Model.list(filter=f'display_name="{model}"')[
        0
    ].resource_name
    # update thresholds to use ThresholdConfig class, if nothing is set, below won't configure
    for i in skew_thresholds:
        skew_thresholds.update({i: ThresholdConfig(value=skew_thresholds[i])})
    for i in attribution_score_skew_thresholds:
        attribution_score_skew_thresholds.update(
            {i: ThresholdConfig(value=attribution_score_skew_thresholds[i])}
        )
    for i in drift_thresholds:
        drift_thresholds.update({i: ThresholdConfig(value=drift_thresholds[i])})
    if default_skew_threshold > 0:
        default_skew_threshold = ThresholdConfig(value=default_skew_threshold)
    batch_prediction_job = BatchPredictionJob(
        display_name=job_display_name,
        model=model_resource_name,
        labels=labels,
        encryption_spec=EncryptionSpec(kms_key_name=encryption_spec_key_name),
        input_config=BatchPredictionJob.InputConfig(
            instances_format=instances_format,
            bigquery_source=BigQuerySource(input_uri=bigquery_source_input_uri),
        ),
        instance_config=BatchPredictionJob.InstanceConfig(
            instance_type=instance_type,
            excluded_fields=excluded_fields,
        ),
        output_config=BatchPredictionJob.OutputConfig(
            predictions_format=predictions_format,
            bigquery_destination=BigQueryDestination(
                output_uri=bigquery_destination_output_uri
            ),
        ),
        dedicated_resources=BatchDedicatedResources(
            machine_spec=MachineSpec(machine_type=machine_type),
            starting_replica_count=1,
            max_replica_count=1,
        ),
        # Model monitoring service will be triggerred if provide following configs.
        model_monitoring_config=ModelMonitoringConfig(
            alert_config=ModelMonitoringAlertConfig(
                email_alert_config=ModelMonitoringAlertConfig.EmailAlertConfig(
                    user_emails=user_emails
                )
            ),
            objective_configs=[
                ModelMonitoringObjectiveConfig(
                    training_dataset=ModelMonitoringObjectiveConfig.TrainingDataset(
                        bigquery_source=BigQuerySource(input_uri=train_data_uri),
                        target_field=target_column_name,
                    ),
                    training_prediction_skew_detection_config=ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig(
                        skew_thresholds=skew_thresholds,
                        attribution_score_skew_thresholds=attribution_score_skew_thresholds,
                        default_skew_threshold=default_skew_threshold,
                    ),
                    # TODO: check drift monitoring works correctly, currently we are testing with training data and drift never gets triggered
                    prediction_drift_detection_config=ModelMonitoringObjectiveConfig.PredictionDriftDetectionConfig(
                        drift_thresholds=drift_thresholds
                    ),
                )
            ],
            stats_anomalies_base_directory=GcsDestination(
                output_uri_prefix=monitor_output_uri
            ),
        ),
        generate_explanation=True,
    )
    client = JobServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )
    out = client.create_batch_prediction_job(
        parent=f"projects/{project}/locations/{location}",
        batch_prediction_job=batch_prediction_job,
    )
    batch_job_id = out.name.split("/")[-1]
    stats_gcs_folder = monitor_output_uri + "/job-" + batch_job_id + "/bp_monitoring/"
    skew_gcs_folder = (
        stats_gcs_folder
        + "stats_and_anomalies/anomalies/training_prediction_skew_anomalies"
    )
    # Wait till the job finish
    name = client.batch_prediction_job_path(
        project=project, location=location, batch_prediction_job=batch_job_id
    )
    response = client.get_batch_prediction_job(name=name)
    print("Job state: ", str(response.state).split(".")[-1])
    while str(response.state) == "JobState.JOB_STATE_PENDING":
        response = client.get_batch_prediction_job(name=name)
        time.sleep(1)
    print("Job state: ", str(response.state).split(".")[-1])
    while str(response.state) == "JobState.JOB_STATE_RUNNING":
        response = client.get_batch_prediction_job(name=name)
        time.sleep(1)
    print("Job state: ", str(response.state).split(".")[-1])
    print("Batch prediction job id: ", batch_job_id)
    print("Stats gcs folder: ", stats_gcs_folder)
    print("Skew gcs folder: ", skew_gcs_folder)
    return (batch_job_id, stats_gcs_folder, skew_gcs_folder, bigquery_destination_output_uri)


@component(
    packages_to_install=["google-cloud-bigquery", "gcsfs"],
    base_image="python:3.7",
)
def monitor_explanation_result_writer(
    project: str,
    dataset_name: str,
    conduit_prediction_storage_table: str,
    monitor_explanation_dest_table: str,
    skew_gcs_folder: str,
    encryption_spec_key_name: str,
    labels: dict,
    model_name: str,
    entity_id: str,
    execution_date: str,
    enable_explanation: bool,
    enable_drift_monitoring: bool,
):
    """Updates batch prediction drift/skew monitoring job result to specified conduit BQ table.

    Args:
        project:
            Project to retrieve endpoint from. If not set, project
            set in aiplatform.init will be used
        dataset_name:
            The name of the dataset in the project.
        conduit_prediction_storage_table:
            conduit prediction result BQ table
        monitor_explanation_dest_table:
            Batch prediction monitoring and explanation result BQ table
        skew_gcs_folder:
            GCS drift/skew result stored location from the batch prediction monitoring task
        encryption_spec_key_name: 
            KMS key for encryption
            Example: projects/{project-id}/locations/{locations}/keyRings/{locations}-nprod-hsm/cryptoKeys/{env}
        labels:
            Dict of labels for cost tracking
        model_name:
            Name of model in Model Registry
            Example: "mymodel"
        entity_id:
            Entity ID column name
            Example: "member_id"
        execution_date:
            Date prediction is executed.
            Example: "1900-01-02"
        enable_explanation:
            Switch for model explanation in batch prediction pipeline.
            Example: True
        enable_drift_monitoring:
            Switch for drift monitoring in batch prediction pipeline.
            Example: True
    """
    from google.cloud import bigquery
    from google.cloud.bigquery.job.query import QueryJobConfig
    import gcsfs
    import time
    
    # An alternative is creating a dsl.Condition() block for this component
    # The downside is that it requires enable_explanation and enable_drift_monitoring
    # be passed as pipeline argument to be used in dsl.Condition(), the same as
    # how task_type is used to trigger different pipeline.
    if not enable_explanation and not enable_drift_monitoring:
        return
    
    # Create Bigquery client
    client = bigquery.Client(project=project)
    client.encryption_configuration = bigquery.EncryptionConfiguration(
        kms_key_name=encryption_spec_key_name
    )
    job_config = QueryJobConfig(labels=labels)
    
    # Get the results of skew drift monitoring if turned on
    if enable_drift_monitoring:
        fs = gcsfs.GCSFileSystem(project=project)
        print("read from gcs dir:", skew_gcs_folder)
        while fs.exists(skew_gcs_folder) == False:
            time.sleep(2)
        skew_drift_stat = fs.open(skew_gcs_folder).read().decode("utf-8")
        skew_drift_result = skew_drift_stat[skew_drift_stat.rfind("anomaly_info") :].replace("\n", "")
    
    # Update explanation and/or drift monitoring results
    set_list = []
    if enable_explanation:
        set_list.append("explanation = TO_JSON(T.explanation)")
    if enable_drift_monitoring:
        set_list.append(f"feature_skew = TO_JSON('{skew_drift_result}')")
    set_string = ",".join(set_list)
    query = f"""
        UPDATE
          `{conduit_prediction_storage_table}` P
        SET
          {set_string}
        FROM
          (
            SELECT
              {entity_id},
              MAX(explanation) AS explanation
            FROM
              `{monitor_explanation_dest_table}`
            GROUP BY
              {entity_id}
          ) T
        WHERE
          P.prediction_execution_date = "{execution_date}"
          AND P.model_name = "{model_name}"
          AND P.entity_id = T.{entity_id}
    """
    query_job = client.query(query, job_config)
    result = query_job.result()


def query_generator(
    input_query: str,
    min_date: str,
    max_date: str,
    date_format: Optional[
        str
    ] = "",  # Using none by default throws an type error by kfp
) -> NamedTuple("Outputs", [("formatted_query", str)]):
    """Create a complete SQL by inserting values into SQL template

    Args:
        input_query: SQL query template
            Example: "SELECT * FROM a WHERE date BETWEEN '{min_date}' AND '{max_date}'"
        min_date: Lower bound of date window
            Example: "1900-01-01"
        max_date: Upper bound of date window
            Exapmle: "1900-01-02"
        date_format: Specify date format to be used in SQL (optional)
            Example: "%Y%m%d"

    Returns:
        formatted_query: A complete SQL
            Example: "SELECT * FROM a WHERE date BETWEEN '1900-01-01' AND '1900-01-02'"
    """
    from datetime import datetime

    if date_format != "":
        min_date = datetime.strptime(min_date, "%Y-%m-%d").strftime(date_format)
        max_date = datetime.strptime(max_date, "%Y-%m-%d").strftime(date_format)
    formatted_query = input_query.format(
        min_date=min_date,
        max_date=max_date,
    )
    return (formatted_query,)


def data_exporter(
    formatted_sql: str,
    bigquery_source_input_uri: str,
    project: str,
    labels: Dict[str, str],
    encryption_spec_key_name: str,
) -> NamedTuple("Outputs", [("input_table_paths", str)]):
    """Load input for batch prediction and export to GCS bucket

    Args:
        formatted_sql: SQL query to load input for batch prediction
            Example: "SELECT * FROM a WHERE date BETWEEN '1900-01-01' AND '1900-01-02'"
        export_path: GCS file path to export data. Must use star (*) in file name
            Example: "gs://bucket/dir/batch-pred-input-*.parquet"
        project: GCP project
            Exapmle: "anbc-dev"
        labels: Labels to track cost
            Example: {"contact": "xyz"}
        encryption_spec_key_name: KMS key for encryption
            Example: projects/{project-id}/locations/{locations}/keyRings/{locations}-nprod-hsm/cryptoKeys/{env}

    Returns:
        input_file_paths: List of file input file names
            Example: ["gs://bucket/dir/batch-pred-input-001.parquet"]
    """
    from time import time
    
    from google.cloud import bigquery
    from google.cloud.bigquery.job.query import QueryJobConfig
    from google.cloud import storage

    # Append export data options to query
    print("Inserting query results into a BQ table...")
    now_str = str(time()).replace(".", "")
    bq_temp_table = f"{bigquery_source_input_uri}--{now_str}"
    create_table = f"""
        CREATE TABLE `{bq_temp_table}`
        AS
        {formatted_sql}
    """
    print("Query:", create_table)
    # Execute query
    client = bigquery.Client(project=project)
    client.encryption_configuration = bigquery.EncryptionConfiguration(
        kms_key_name=encryption_spec_key_name
    )
    job_config = QueryJobConfig(labels=labels)
    query_job = client.query(create_table, job_config)
    # Wait until job is done
    query_job.result()
    return (bq_temp_table,)



def prediction_generator_temp(
    bigquery_source_input_uri: str,
    bigquery_destination_output_uri: str,
    entity_id: str,
    endpoint_name: str,
    model_name: str,
    project: str,
    location: str,
    encryption_spec_key_name: str,
    feature_names: List[str],
    labels: Dict[str, str]
) -> NamedTuple(
    "Outputs",
    [
        ("bigquery_destination_output_uri", str),
        ("model_version_id", str),
        ("model_id", str),
    ],
):
    """Generate predictions and export to GCS path

    Args:
        pred_data_paths: List of file input file names
            Example: ["gs://bucket/dir/batch-pred-input-001.parquet"]
        entity_id: Entity ID column name
            Example: "member_id"
        endpoint_name: Name of endpoint
            Exapmle: "myendpoint"
        model_name: Name of model in Model Registry
            Example: "mymodel"
        gcs_output_dir: GCS dir path to output predictions
            Example: "gs://bucket/dir/path"
        project: GCP project
            Example: "anbc-dev"
        location: GCP region
            Example: "us-east4"
        encryption_spec_key_name: KMS key for encryption
            Example: projects/{project-id}/locations/{locations}/keyRings/{locations}-nprod-hsm/cryptoKeys/{env}
        feature_names: List of feature names
            Example: ["os", "is_mobile"]
        prediction_execution_date: Date prediction is executed
            Example: "1900-01-02"

    Returns:
        output_path: GCS path to output predictions
            Example: "gs://bucket/dir/batch-pred.parquet"
    """
    import os
    from datetime import datetime, timezone
    from time import time

    from google.cloud import aiplatform
    from google.cloud import bigquery
    import pandas as pd
    from tqdm import tqdm

    # Load input
    print("Loading input data...")
    select_prediction_input = f"""
        SELECT * FROM `{bigquery_source_input_uri}`
    """
    print("Loading predictions...")
    client = bigquery.Client(project=project)
    client.encryption_configuration = bigquery.EncryptionConfiguration(
        kms_key_name=encryption_spec_key_name
    )
    job_config = bigquery.QueryJobConfig(labels=labels)
    pred_df = client.query(select_prediction_input, job_config).to_dataframe()
    # endpoint_input_df = pred_df.iloc[:, ~pred_df.columns.isin([entity_id])]
    endpoint_input_df = pred_df[feature_names]
    # Divide input list into chunks to avoid input data limit error (1.5MB max)
    print(f"Input shape = {endpoint_input_df.shape}")
    print("Generating predictions...")
    aiplatform.init(project=project, location=location)
    endpoint = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"',
        project=project,
        location=location,
    )[0]
    input_list = endpoint_input_df.values.tolist()
    n = 500
    input_chunks = [input_list[i : i + n] for i in range(0, len(input_list), n)]
    predictions = []
    print("Num input chunks:", len(input_chunks))
    for c in tqdm(input_chunks):
        result = endpoint.predict(c)
        predictions.extend([x for x in result.predictions])
    # predictions = endpoint.predict(input_list).predictions
    # Prepare output
    print("Preparing output...")
    output_df = pd.DataFrame()
    # TODO: Let Joey know about the requirements: y_pred expects label prediction, probability should be stored in y_proba if any
    output_df["y_pred"] = [1 if x > 0.1 else 0 for x in predictions]
    output_df["entity_id"] = pred_df[entity_id]
    # NOTE: write df as bq table to make it compatible with Batch Prediction service
    # Create temp table
    now_str = str(time()).replace(".", "")
    bq_temp_table = f"{bigquery_destination_output_uri}-{now_str}"
    create_table = f"""
        CREATE TABLE `{bq_temp_table}` (
            y_pred FLOAT64,
            entity_id STRING
        );
    """
    query_job = client.query(create_table, job_config)
    query_job.result()
    # Insert rows into temp table
    rows_to_insert = output_df.to_dict("records")
    errors = client.insert_rows_json(bq_temp_table, rows_to_insert)
    if len(errors) == 0:
        print("New rows have been added")
    else:
        raise RuntimeError("Encountered errors while inserting rows: {}".format(errors))
    print("Done!")
    return (bq_temp_table, str(result.model_version_id), str(result.model_resource_name.split("/")[-1]))



def prediction_writer(
    prediction_window_sec: int,
    project: str,
    model_version_id: str,
    model_name: str,
    model_id: str,
    prediction_type: str,
    prediction_execution_date: str,
    bq_output_table_path: str,
    bq_prediction_storage_uri: str,
    encryption_spec_key_name: str,
    labels: Dict[str, str],
) -> None:
    """Write prediciton to Prediction storage

    Args:
        prediction_results_path: GCS path to predictions
            Example: "gs://bucket/dir/pred.parquet"
        prediction_window_sec: Specify how long prediction will be valid in seconds
            Example: 86400
        project: GCP project
            Example: "anbc-dev"
        bq_prediction_storage_uri: BQ URI to preidciton storage
            Example: "project.dataset.table"
        encryption_spec_key_name: KMS key for encryption
            Example: projects/{project-id}/locations/{locations}/keyRings/{locations}-nprod-hsm/cryptoKeys/{env}
    """
    from datetime import datetime 
    import subprocess
    import sys
    import warnings
    
    from google.cloud import bigquery
    from google.cloud.bigquery.job.query import QueryJobConfig
    from google.cloud import bigquery_storage_v1
    from google.cloud.bigquery_storage_v1 import types
    from google.cloud.bigquery_storage_v1 import writer
    from google.protobuf import descriptor_pb2
    import pandas as pd

    # Refer to 'prediction_generator' for DF schema
    select_predictions = f"""
        SELECT * FROM `{bq_output_table_path}`
    """
    print("Loading predictions...")
    client = bigquery.Client(project=project)
    client.encryption_configuration = bigquery.EncryptionConfiguration(
        kms_key_name=encryption_spec_key_name
    )
    job_config = QueryJobConfig(labels=labels)
    output_df = client.query(select_predictions, job_config).to_dataframe()
    # Format prediction execution date
    prediction_execution_date = datetime.strptime(
        prediction_execution_date, "%Y-%m-%d"
    ).strftime("%Y-%m-%d %H:%M:%S")
    # Write records using BQ storage write API
    # Write proto file
    print("Writing proto file...")
    proto_file_name = "conduit_prediction.proto"
    # TODO: use enum for prediction_type
    # TODO: use repeated for all fields except for y_pred and entity_id (?) --> need more research
    proto = """
        syntax = "proto2";

        message ConduitPrediction {
            optional double y_pred = 1;
            optional string prediction_execution_date = 2;
            optional int64 prediction_window_sec = 3;
            optional string entity_id = 4;
            optional string prediction_type = 5; 
            optional string model_version_id = 6;
            optional string model_name = 7;
            optional string model_id = 8;
        }
    """
    with open(proto_file_name, "w") as f:
        f.write(proto)
    # Compile proto file
    print("Compiling proto file...")
    command = f"protoc --python_out=. {proto_file_name}"
    result = subprocess.run(
        command, 
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print("Stdout:", result.stdout.decode("utf-8"))
        print("Stderr:", result.stderr.decode("utf-8"))
        raise RuntimeError(f"Command ('{command}') failed with returncode = {result.returncode}")
    # Define function to create proto row
    import conduit_prediction_pb2

    def create_row_data(
        y_pred: float,
        prediction_execution_date: str,
        prediction_window_sec: int,
        entity_id: str,
        prediction_type: str,
        model_version_id: str,
        model_name: str,
        model_id: str,
    ):
        row = conduit_prediction_pb2.ConduitPrediction()
        row.y_pred = y_pred
        row.prediction_execution_date = prediction_execution_date
        row.prediction_window_sec = prediction_window_sec
        row.entity_id = entity_id
        row.prediction_type = prediction_type
        row.model_version_id = model_version_id
        row.model_name = model_name
        row.model_id = model_id
        return row.SerializeToString()
    
    # Create write stream
    print("Creating write stream...")
    write_client = bigquery_storage_v1.BigQueryWriteClient()
    bq_uri = bq_prediction_storage_uri.split(".")
    parent = write_client.table_path(bq_uri[0], bq_uri[1], bq_uri[2])
    write_stream = types.WriteStream()
    write_stream.type_ = types.WriteStream.Type.PENDING
    write_stream = write_client.create_write_stream(
        parent=parent, write_stream=write_stream
    )
    stream_name = write_stream.name
    print("Write stream name:", write_stream.name)
    # Create append rows request
    request_template = types.AppendRowsRequest()
    request_template.write_stream = stream_name
    # Define schema
    print("Defining schema...")
    proto_schema = types.ProtoSchema()
    proto_descriptor = descriptor_pb2.DescriptorProto()
    conduit_prediction_pb2.ConduitPrediction.DESCRIPTOR.CopyToProto(proto_descriptor)
    proto_schema.proto_descriptor = proto_descriptor
    proto_data = types.AppendRowsRequest.ProtoData()
    proto_data.writer_schema = proto_schema
    request_template.proto_rows = proto_data
    # Append rows to write stream
    append_rows_stream = writer.AppendRowsStream(write_client, request_template)
    # Create rows
    print("Creating proto rows...")
    proto_rows = types.ProtoRows()
    # NOTE: Drop duplicate rows as they cause '400 UPDATE/MERGE must match at most one source row for each target row' in backfiller
    num_rows_before_drop_duplicate = output_df.shape[0]
    output_df = output_df.drop_duplicates(["entity_id"])
    if num_rows_before_drop_duplicate > output_df.shape[0]:
        warnings.warn(f"{num_rows_before_drop_duplicate - output_df.shape[0]} duplicate entity_ids found in batch prediction query. Dropping duplicates...")
    for row in output_df.to_dict("records"):
        proto_rows.serialized_rows.append(create_row_data(
            y_pred=row["y_pred"],
            prediction_execution_date=prediction_execution_date,
            prediction_window_sec=prediction_window_sec,
            entity_id=row["entity_id"],
            prediction_type=prediction_type,
            model_version_id=model_version_id,
            model_name=model_name,
            model_id=model_id
        ))
    # Add rows to append request
    request = types.AppendRowsRequest()
    request.offset = 0
    proto_data = types.AppendRowsRequest.ProtoData()
    proto_data.rows = proto_rows
    request.proto_rows = proto_data
    # Send request to write stream
    print("Sending rows to write stream...")
    response_future_1 = append_rows_stream.send(request)
    result = response_future_1.result()
    print("Append rows request result:", str(result))
    append_rows_stream.close()
    # Commit write stream
    print("Commiting write stream...")
    write_client.finalize_write_stream(name=write_stream.name)
    batch_commit_write_streams_request = types.BatchCommitWriteStreamsRequest()
    batch_commit_write_streams_request.parent = parent
    batch_commit_write_streams_request.write_streams = [write_stream.name]
    result = write_client.batch_commit_write_streams(batch_commit_write_streams_request)
    print("Commit result:", str(result))
    print("Done!")


# NOTE: if base is none, today becomes base_date. Add base_date argument makes debugging/dev easier
def run_config_generator(
    window_len_day: int, base_date_str: Optional[str] = None
) -> NamedTuple("Outputs", [("min_date", str), ("max_date", str)]):
    """Generate run time configuration

    Args:
        window_len_day: Date window length: min_date = max_date - window_len_day
            Example: 1
        base_date_str: Specify the base date (mainly for dedbugging). If not specified, today is used for the base date
            Example: "1900-01-01"

    Returns:
        min_date: Lower bound of date window
            Example: "1900-01-01"
        max_date: Upper bound of date window
            Example: "1900-01-02"
    """
    from datetime import datetime, timedelta

    if base_date_str is None:
        base_date = datetime.today()
    else:
        base_date = datetime.strptime(base_date_str, "%Y-%m-%d")
    # TODO: check with Viren whether we need to get date string format from the end user
    min_date = (base_date - timedelta(days=window_len_day)).strftime("%Y-%m-%d")
    max_date = base_date.strftime("%Y-%m-%d")
    return (min_date, max_date)


# Components for OOS Evaluation


def metric_computer(
    gt_pred_pair_path: str,
    target_metrics: List[str],
) -> NamedTuple("Outputs", [("metrics_result", dict)]):
    """Compute evaluation metrics

    Args:
        gt_pred_pair_path: GCS path to ground truth and prediction pairs
            Example: "gs://bucket/dir/gt_pred_pair.parquet"
        target_metrics: List of metrics to compute
            Example: ["accuracy", "recall"]

    Returns:
        metrics_result: Key value pair where key is metrics and value is metric value
            Example: {"accuracy": 0.8, "recall": 0.2}
    """
    from typing import Callable

    import pandas as pd

    # TODO: Create a module for this function so it can be reused in training pipeline
    def get_metric_fn(metric: str) -> Callable:
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score

        if metric == "accuracy":
            return accuracy_score
        elif metric == "precision":
            return precision_score
        elif metric == "recall":
            return recall_score
        elif metric == "f1":
            return f1_score
        else:
            raise RuntimeError(f"Unknown metric name: {metric}")

    def load_parquet(gcs_path: str) -> pd.DataFrame:
        return pd.read_parquet(gcs_path)

    gt_pred_pair = load_parquet(gt_pred_pair_path)
    metrics_result = {}
    for metric_name in target_metrics:
        metric_fn = get_metric_fn(metric_name)
        metrics_result[metric_name] = metric_fn(
            gt_pred_pair["y_actual"].astype(float), gt_pred_pair["y_pred"].astype(float)
        )

    return (metrics_result,)


def oos_metrics_result_writer(
    oos_metrics_result: dict,
    bq_train_metric_storage_uri: str,
    bq_oos_metric_storage_uri: str,
    model_id: str,
    model_version_id: str,
    model_name: str,
    project: str,
    labels: Dict[str, str],
    encryption_spec_key_name: str,
) -> None:
    """Write OOS metrics to OOS metric storage

    Args:
        bq_train_metric_storage_uri: BQ URI to train storage
            Example: "project.dataset.table"
        bq_oos_metric_storage_uri: BQ URI to OOS metric storage
            Example: "project.dataset.table"
        model_id: ID of model
            Example: "1234"
        model_version_id: Version ID of model
            Example: "1"
        model_name: Name of model
            Example: "mymodel"
        project: GCP project
            Example: "anbc-dev"
        labels: Labels to track costs
            Example: {"contact": "1234"}
    """
    import warnings
    import json

    from google.cloud import bigquery
    from datetime import datetime, timezone

    # Need to pass project; otherwise, it gets 403 Permission Error
    client = bigquery.Client(project=project)
    client.encryption_configuration = bigquery.EncryptionConfiguration(
        kms_key_name=encryption_spec_key_name
    )
    job_config = bigquery.QueryJobConfig(labels=labels)
    # Get baseline metrics
    print("Loading train metrics...")
    get_train_metrics = f"""
        SELECT 
            metrics -- JSON output
        FROM `{bq_train_metric_storage_uri}`
        WHERE model_id = '{model_id}' AND model_version_id = '{model_version_id}'
    """
    print("Query:", get_train_metrics)
    query_job = client.query(get_train_metrics, job_config)
    select_result = [row for row in query_job.result()]
    # TODO: Use logger.debug (Need to know which logging module to use)
    print(f"Query result: {select_result}")
    # Extract rows without column name
    select_result = [[x for x in dict(row).values()][0] for row in query_job.result()]
    if len(select_result) == 0:
        raise RuntimeError(
            f"No train metrics found for model_id = '{model_id}' and model_version_id = '{model_version_id}'"
        )
    elif len(select_result) != 1:
        warnings.warn(
            f"Found more than one train metrics associated with model_id = '{model_id}' and model_version_id = '{model_version_id}'. This may lead to an unexpected behavior. Please make sure that there is no duplicate records in 'Train Storage'"
        )
    # JSON string
    train_metrics: str = select_result[0]
    # If dict is empty, there is no baseline metrics associated with the model
    # Write to metric storage
    # TODO: store datetime format str as constant in the package
    print("Writing to OOS metric storage...")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows_to_insert = [
        {
            "model_id": model_id,
            "model_version_id": model_version_id,
            "model_name": model_name,
            "timestamp": timestamp,
            "train_metrics": train_metrics,
            "oos_metrics": json.dumps(oos_metrics_result),
        }
    ]
    print(rows_to_insert)
    errors = client.insert_rows_json(bq_oos_metric_storage_uri, rows_to_insert)
    if len(errors) == 0:
        print("New rows have been added")
    else:
        raise RuntimeError("Encountered errors while inserting rows: {}".format(errors))
    print("Done!")


# NOTE: if base is none, today becomes base_date. Add base_date argument makes debugging/dev easier
# NOTE: temp name
def run_config_generator_oos(
    schedule_frequency_day: int, base_date_str: Optional[str] = None
) -> NamedTuple("Outputs", [("min_date", str), ("max_date", str)]):
    """Generate run time configuration

    Args:
        schedule_frequency_day: Frequency to run pipeline in days: min_date = max_date - schedule_frequency_day
            Example: 1
        base_date_str: Specify the base date (mainly for dedbugging). If not specified, today is used for the base date
            Example: "1900-01-01"

    Returns:
        min_date: Lower bound of date window
            Example: "1900-01-01"
        max_date: Upper bound of date window
            Example: "1900-01-02"
    """
    from datetime import datetime, timedelta

    if base_date_str is None:
        base_date = datetime.today()
    else:
        base_date = datetime.strptime(base_date_str, "%Y-%m-%d")
    # TODO: check with Viren whether we need to get date string format from the end user
    min_date = (base_date - timedelta(days=schedule_frequency_day)).strftime("%Y-%m-%d")
    max_date = base_date.strftime("%Y-%m-%d")
    return (min_date, max_date)


# NOTE: temp name
def query_generator_oos(
    input_query: str,
    min_date: str,
    max_date: str,
    date_format: Optional[str] = "",
) -> NamedTuple("Outputs", [("formatted_query", str)]):
    """Create a complete SQL by inserting values into SQL template

    Args:
        input_query: SQL query template
            Example: "SELECT * FROM a WHERE date BETWEEN '{min_date}' AND '{max_date}'"
        min_date: Lower bound of date window
            Example: "1900-01-01"
        max_date: Upper bound of date window
            Exapmle: "1900-01-02"
        date_format: Specify date format to be used in SQL (optional)
            Example: "%Y%m%d"

    Returns:
        formatted_query: A complete SQL
            Example: "SELECT * FROM a WHERE date BETWEEN '1900-01-01' AND '1900-01-02'"
    """
    from datetime import datetime

    if date_format != "":
        min_date = datetime.strptime(min_date, "%Y-%m-%d").strftime(date_format)
        max_date = datetime.strptime(max_date, "%Y-%m-%d").strftime(date_format)
    formatted_query = input_query.format(
        min_date=min_date,
        max_date=max_date,
    )
    return (formatted_query,)


def prediction_ground_truth_matcher(
    ground_truth_query: str,
    ground_truth_min_date: str,
    ground_truth_max_date: str,
    model_name: str,
    project: str,
    entity_id: str,
    gcs_output_dir: str,
    labels: Dict[str, str],
    bq_prediction_storage_uri: str,
    encryption_spec_key_name: str,
) -> NamedTuple(
    "Outputs",
    [("gt_pred_pair_path", str), ("model_id", str), ("model_version_id", str)],
):
    """Match ground truths and predictions in tables and export to GCS path

    Args:
        ground_truth_query: A SQL query to load ground truths
            Example: "SELECT * FROM a WHERE date BETWEEN '1900-01-01' AND '1900-01-02'"
        ground_truth_min_date: Min date for ground truth query
            Example: "1900-01-01"
        ground_truth_max_date: Max date for ground truth query
            Example: "1900-01-02"
        model_name: Name of model
            Example: "mymodel"
        project: GCP project
            Example: "anbc-dev"
        entity_id: Name of entity id column
            Example: "member_id"
        gcs_outptu_dir: GCS dir path to output prediction ground truths pairs
            Example: "gs://bucket/dir/data/"
        labels: Labels for tracking costs
            Example: {"contact": "xyz"}
        bq_prediction_storage_uri: BQ URI to prediction storage
            Example: "project.dataset.table"

    Returns:
        gt_pred_pair_path: GCS path to ground truths prediction pairs
            Example: "gs://bucket/dir/data/gt_pred_pairs.parquet"
        model_id: ID of model
            Example: "1234"
        model_version_id: Version ID of model
            Example: "1234"
    """
    import os
    import warnings 
    
    from google.cloud.bigquery.job.query import QueryJobConfig
    from google.cloud import bigquery
    import pandas as pd

    # Load gt
    print("Loading ground truth...")
    job_config = QueryJobConfig(labels=labels)
    client = bigquery.Client(project=project)
    client.encryption_configuration = bigquery.EncryptionConfiguration(
        kms_key_name=encryption_spec_key_name
    )
    ground_truth_df = client.query(ground_truth_query, job_config).to_dataframe()
    # Load predictions
    print("Loading predictions...")
    get_predictions = f"""
        SELECT y_pred,
               entity_id,
               prediction_execution_date,
               model_id,
               model_version_id,
               model_name
        FROM   `{bq_prediction_storage_uri}`
        WHERE  model_name = '{model_name}'
        AND    CAST('{ground_truth_min_date}' AS TIMESTAMP) <= prediction_execution_date
        AND    TIMESTAMP_ADD(prediction_execution_date, INTERVAL prediction_window_sec SECOND ) <= CAST('{ground_truth_max_date}' AS TIMESTAMP) -- TODO: store prediction_execution_date + prediction_window_sec in prediction_writer to save cost. Executing TIMESTAMP_ADD adds additional cost
    """
    prediction_df = client.query(get_predictions, job_config).to_dataframe()
    print("Query:", get_predictions)
    if prediction_df.shape[0] == 0:
        raise RuntimeError("Unable to find predictions. Check query")
    # Join ground truth and predictions
    print("Joining ground truths and predictions...")
    # Rename columns for join
    ground_truth_df.rename(columns={entity_id: "entity_id"}, inplace=True)
    # Explicitly convert entity_id to str to avoid type mismatch error when join
    ground_truth_df["entity_id"] = ground_truth_df["entity_id"].astype(str)
    prediction_df["entity_id"] = prediction_df["entity_id"].astype(str)
    # NOTE: Drop duplicate rows as they cause '400 UPDATE/MERGE must match at most one source row for each target row' in backfiller
    num_rows_before_drop_duplicate = ground_truth_df.shape[0]
    ground_truth_df = ground_truth_df.drop_duplicates(["entity_id"])
    if num_rows_before_drop_duplicate > ground_truth_df.shape[0]:
        warnings.warn(f"{num_rows_before_drop_duplicate - ground_truth_df.shape[0]} duplicate columns found in ground truths query. Dropping duplicates...")
    pred_gt_pairs_df = prediction_df.merge(
        ground_truth_df.drop_duplicates(["entity_id"]), on=["entity_id"], how="left"
    )
    pred_gt_pairs_df["y_actual"] = pred_gt_pairs_df["y_actual"].fillna(0)
    pred_gt_pairs_df["y_pred"] = pred_gt_pairs_df.y_pred
    pred_gt_pairs_df["model_name"] = pred_gt_pairs_df.model_name
    pred_gt_pairs_df[
        "prediction_execution_date"
    ] = pred_gt_pairs_df.prediction_execution_date.astype(str)
    # Write gt-pred pairs to GCS
    print("Writing pairs to GCS bucket...")
    gcs_output_path = os.path.join(
        gcs_output_dir,
        "pred_gt_pairs.parquet",
    )
    pred_gt_pairs_df[
        [
            "y_actual",
            "y_pred",
            "entity_id",
            "prediction_execution_date",
            "model_name",
        ]
    ].to_parquet(gcs_output_path, index=False)
    print("Done!")
    # Model id and model version id should be the same for all rows
    model_id = str(prediction_df["model_id"][0])
    model_version_id = str(prediction_df["model_version_id"][0])
    return (str(gcs_output_path), model_id, model_version_id)


def ground_truth_backfiller(
    gt_pred_pair_path: str,
    labels: Dict[str, str],
    bq_prediction_storage_uri: str,
    project: str,
    bq_dataset: str,
    model_name: str,
) -> None:
    """Backfill y_actual into prediction storage. It's an individual component since update operation is not very straightforward in BQ (BQ is primarily designed to be append-only)."""
    from time import time, sleep

    from google.cloud import bigquery
    from google.api_core.exceptions import BadRequest
    import pandas as pd

    # Create temp table to store gt-pred pairs
    print("Creating temp BQ table...")
    now_str = str(time()).replace(".", "")
    bq_temp_table = f"{project}.{bq_dataset}.{model_name}-{now_str}"
    print("Temp table:", bq_temp_table)
    create_temp_table = f"""
        CREATE TABLE `{bq_temp_table}` (
            y_pred FLOAT64,
            y_actual FLOAT64,
            entity_id STRING,
            prediction_execution_date TIMESTAMP,
            model_name STRING
        );
    """
    job_config = bigquery.QueryJobConfig(labels=labels)
    client = bigquery.Client(project=project)
    query_job = client.query(create_temp_table, job_config)
    query_job.result()
    try:
        # Insert rows to temp table
        print("Inserting gt-pred pairs into temp table...")
        gt_pred_pair_df = pd.read_parquet(gt_pred_pair_path)
        rows_to_insert = gt_pred_pair_df.to_dict("records")
        errors = client.insert_rows_json(bq_temp_table, rows_to_insert)
        if len(errors) == 0:
            print("New rows have been added")
        else:
            raise RuntimeError(
                "Encountered errors while inserting rows: {}".format(errors)
            )
        # Bulk update y_actual in prediction storage
        backfill_y_actual = f"""
            UPDATE `{bq_prediction_storage_uri}` PS
            SET PS.y_actual = TEMP.y_actual
            FROM (
                SELECT * FROM `{bq_temp_table}`
            ) TEMP
                WHERE PS.entity_id = TEMP.entity_id 
                        AND PS.prediction_execution_date = TEMP.prediction_execution_date
                        AND PS.model_name = TEMP.model_name
        """
        # NOTE: it takes up to 30 minutes for the rows to be available for DML in BQ (usually takes 90 secs with Storage Write API)
        # Therefore, this component attempts to update rows every 90 sec and fails if it doesn't success in 30 min
        # Referecen on wait time: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-manipulation-language#limitations
        print("Updating rows in prediction storage...")
        wait_time_sec = 90
        num_try = 0
        max_try = 20
        while True:
            try:
                query_job = client.query(backfill_y_actual, job_config)
                query_job.result()
                print("Sucessfully update rows!")
                break
            except BadRequest as e:
                print("Encountered the following exception:", str(e), "Retrying...")
            # Pausing process for the next attempt
            num_try += 1
            if num_try >= max_try:
                raise RuntimeError("Maximum attempts reached to update rows")
            print(f"Pausing program for {wait_time_sec} seconds. {num_try}/{max_try} attempt has made")
            sleep(wait_time_sec)
            print("Done!")
    except Exception as e:
        raise RuntimeError(f"Encountered the following exception: {str(e)}")
    finally:
        # Remove temp table
        print("Dropping temp table...")
        drop_temp_table = f"DROP TABLE IF EXISTS `{bq_temp_table}`"
        query_job = client.query(drop_temp_table, job_config)
        query_job.result()


# NOTE: temp component to enable caching
@component(base_image="python:3.7.12")
def output_dir_generator(
    pipeline_root: str, pipeline_job_name: str
) -> NamedTuple("Outputs", [("model_bucket_uri", str), ("model_folder_uri", str)]):
    """Create model output dir basedd on pipeline job name. This component is necessary to enable
    caching between CustomContainerJobRunOp and model_uploader because pipeline job names are created
    every time a job runs. This causes model uploader to check empty directory. Ideally return model
    artifact uri from custom container job component, but CustomContainerJobRunOp doesn't seem to have
    options to do so.
    
    Args:
        pipeline_root: GCS dir path to pipeline root
            Example: "gs://bucket/dir/path"
        pipeline_job_name: Pipeline job name
            Example: "pipeline-run-10-20230425010448"

    Returns:
        model_bucket_uri: GCS dir path to model output dir
            Example: "gs://bucket/dir/pipeline-run-10-20230425010448"
        model_folder_uri: model_bucket_uri + "model"
            Example: "gs://bucket/dir/pipeline-run-10-20230425010448/model"
    """
    import os

    # Dynamically create output path
    # NOTE: Passed to custom training job op
    MODEL_BUCKET_URI = os.path.join(pipeline_root, str(pipeline_job_name))
    # NOTE: Passed to model uploder
    MODEL_FOLDER_URI = os.path.join(MODEL_BUCKET_URI, "model")
    return (MODEL_BUCKET_URI, MODEL_FOLDER_URI)


def create_pipeline(pipeline_config: dict) -> None:
    @dsl.pipeline(
        pipeline_root=pipeline_config["pipeline_root"],
        name=pipeline_config["pipeline_name"],
    )
    def pipeline(task_type: str, create_endpoint: str):
        # Training Pipeline
        with dsl.Condition(task_type == "training", name="training-pipeline"):
            # NOTE: this component is necessary to enable caching in training pipeline.
            # For example, starting from "model-uploader" fails if disable the component below.
            # It's because dsl.PIPELINE_JOB_NAME_PLACEHOLDER gets refreshed when running a new pipeline job if cache is enabled, and artifact_uri points to an empty directory
            generate_output_dir = output_dir_generator(
                pipeline_root=pipeline_config["pipeline_root"],
                pipeline_job_name=dsl.PIPELINE_JOB_NAME_PLACEHOLDER,
            )
            # Prepare training data
            prepare_training_data = training_data_preparator(
                project=pipeline_config["project"],
                location=pipeline_config["location"],
                train_data_sql=pipeline_config["train_data_sql"],
                train_data_gcs_uri=pipeline_config["train_data_gcs_uri"],
                biglake_table_path=pipeline_config["train_data_table_path"],
                biglake_data_format=pipeline_config["biglake_data_format"],
                biglake_connection_name=pipeline_config["biglake_connection_name"],
                labels=pipeline_config["labels"],
            )
            # Create vertex ai dataset
            create_tabular_dataset = TabularDatasetCreateOp(
                project=pipeline_config["project"],
                location=pipeline_config["location"],
                display_name=pipeline_config["train_data_table_name"],
                gcs_source=pipeline_config["train_data_gcs_uri"],
                labels=pipeline_config["labels"],
                encryption_spec_key_name=pipeline_config["encryption_spec_key_name"],
            )
            create_tabular_dataset.after(prepare_training_data)
            # Run custom training
            train = CustomContainerTrainingJobRunOp(
                display_name=pipeline_config["training_job_display_name"],
                container_uri=pipeline_config["training_image_uri"],
                project=pipeline_config["project"],
                location=pipeline_config["location"],
                staging_bucket=pipeline_config["train_staging_bucket_uri"],
                base_output_dir=generate_output_dir.outputs["model_bucket_uri"],
                environment_variables=json.dumps(
                    {
                        "AIP_TRAINING_DATA_URI": pipeline_config[
                            "train_data_table_uri"
                        ],
                        "AIP_TEST_DATA_URI": pipeline_config["test_data_jsonl_uri"],
                        "AIP_MODEL_DIR": str(
                            generate_output_dir.outputs["model_bucket_uri"]
                        ),
                    }
                ),
                labels=json.dumps(pipeline_config["labels"]),
                training_encryption_spec_key_name=pipeline_config["encryption_spec_key_name"],
                model_encryption_spec_key_name=pipeline_config["encryption_spec_key_name"],
            )
            train.after(create_tabular_dataset)
            # Upload model to model registry
            upload_model = model_uploader(
                artifact_uri=generate_output_dir.outputs["model_folder_uri"],
                display_name=pipeline_config["model_name"],
                custom_predict_image_uri=pipeline_config["serving_image_uri"],
                feature_names=pipeline_config["feature_names"],
                labels=pipeline_config["labels"],
                # TODO: confirm whether there is any pattern for version aliases
                version_aliases=["custom-training"],
                project=pipeline_config["project"],
                location=pipeline_config["location"],
                # TODO: get model description from the user (Update master input table)
                version_description=pipeline_config.get("description", "Test"),
                explanation_method=pipeline_config["explanation_method"],
                encryption_spec_key_name=pipeline_config["encryption_spec_key_name"],
            )
            upload_model.after(train)
            # Evaluate uploaded model
            evaluation = model_evaluator(
                eval_metrics=pipeline_config["eval_metrics"],
                aip_test_data_uri=pipeline_config["test_data_jsonl_uri"],
                aip_model_dir=upload_model.outputs["model_dir"],
            )
            # Check evaluation threshold
            decide_deploy_model_or_not = deploy_model_or_not(
                model_resource_name=upload_model.outputs["model_resource_name"],
                model_version_id=upload_model.outputs["model_version_id"],
                metrics_result=evaluation.outputs["metrics"],
                target_metric=pipeline_config["target_metric"],
                threshold=pipeline_config["target_metric_threshold"],
            )
            # Proceed if deploy model is true
            with dsl.Condition(
                decide_deploy_model_or_not.outputs["decision"] == "true",
                name="decide-deploy-model-or-not",
            ):
                with dsl.Condition(
                    create_endpoint == "true", name="create-endpoint-or-not"
                ):
                    # Create endpoint if not exist and deploy model to endpoint
                    deploy_model = model_deployer(
                        model_id=upload_model.outputs["model_id"],
                        endpoint_display_name=pipeline_config["model_endpoint_name"],
                        project=pipeline_config["project"],
                        location=pipeline_config["location"],
                        machine_type=pipeline_config["model_deploy_machine_type"],
                        min_replica_count=pipeline_config[
                            "model_deploy_min_replica_count"
                        ],
                        enable_request_response_logging=pipeline_config[
                            "model_enable_request_response_logging"
                        ],
                        request_response_logging_bq_destination_table=pipeline_config[
                            "model_request_response_logging_bq_destination_table"
                        ],
                        labels=pipeline_config["labels"],
                        encryption_spec_key_name=pipeline_config["encryption_spec_key_name"],
                    )
                    # Create model monitoring
                    monitor_endpoint = model_endpoint_monitor_creator(
                        # TODO: Update model deployer output
                        endpoint_resource_name=deploy_model.outputs[
                            "endpoint_resource_name"
                        ],
                        display_name=pipeline_config[
                            "model_endpoint_monitoring_job_display_name"
                        ],
                        train_dataset_uri=pipeline_config["train_data_table_uri"],
                        target_column_name=pipeline_config["target_column_name"],
                        feature_names=pipeline_config["feature_names"],
                        stats_anomalies_base_directory=pipeline_config[
                            "model_endpoint_monitoring_stats_anomalies_base_directory"
                        ],
                        project=pipeline_config["project"],
                        location=pipeline_config["location"],
                        labels=pipeline_config["labels"],
                        user_emails=pipeline_config.get(
                            "model_endpoint_monitoring_user_emails", []
                        ),  # Optional param - if None, config associated with this param won't be created
                        default_skew_thresholds=pipeline_config["model_endpoint_monitoring_default_skew_thresholds"],
                        skew_thresholds=pipeline_config.get(
                            "model_endpoint_mointoring_skew_thresholds", {}
                        ),  # Optional param - if None, config associated with this param won't be created
                        attrib_skew_thresholds=pipeline_config.get(
                            "model_endpoint_mointoring_attrib_skew_thresholds", {}
                        ),  # Optional param - if None, config associated with this param won't be created
                        sample_rate=pipeline_config.get(
                            "model_endpoint_monitoring_sample_rate", 1.0
                        ),  # Optional param - if None, config associated with this param won't be created
                        monitor_interval=pipeline_config.get(
                            "model_endpoint_monitoring_monitor_interval_hrs", 24
                        ),  # Optional param - if None, config associated with this param won't be created
                        encryption_spec_key_name=pipeline_config["encryption_spec_key_name"],
                    )
                # Write metrics to train storage
                write_train_metrics = train_metrics_result_writer(
                    metrics_result=evaluation.outputs["metrics"],
                    model_name=pipeline_config["model_name"],
                    model_id=upload_model.outputs["model_id"],
                    model_version_id=upload_model.outputs["model_version_id"],
                    project=pipeline_config["project"],
                    bq_train_metric_storage_uri="anbc-pdev.mleng_platform_ent_pdev.conduit-train-storage",
                )

        # Batch Prediction Pipeline
        with dsl.Condition(
            task_type == "batch_prediction", name="batch-prediction-pipeline"
        ):
            # Get values to create SQL
            get_run_config = component(
                run_config_generator, base_image="python:3.7.12"
            )(
                window_len_day=pipeline_config["window_len_day"],
                base_date_str="2016-08-09",  # NOTE: hard code base date for debugging. in production code, this argument should not be passed
            )
            # Generate input query
            generate_query = component(query_generator, base_image="python:3.7.12")(
                input_query=pipeline_config["batch_prediction_data_sql"],
                min_date=get_run_config.outputs["min_date"],
                max_date=get_run_config.outputs["max_date"],
                date_format=pipeline_config.get("date_format", ""),
            )
            # Export data to gsc bucket
            export_data = component(
                data_exporter,
                base_image="python:3.7.12",
                packages_to_install=[
                    "google-cloud-bigquery",
                    "google-cloud-aiplatform",
                    "gcsfs",
                ],
            )(
                formatted_sql=generate_query.outputs["formatted_query"],
                # TODO: use pipeline_config for export_path (update master input table)
                bigquery_source_input_uri=pipeline_config["batch_prediction_test_data_table_uri"],
                project="anbc-pdev",
                labels=pipeline_config["labels"],
                encryption_spec_key_name=pipeline_config["encryption_spec_key_name"],
            )
            # Actual batch prediction operation:
            batchpredict_op = prediction_generator(
                project=pipeline_config["project"],
                location=pipeline_config["location"],
                job_display_name=pipeline_config["batch_prediction_job_display_name"],
                model=pipeline_config["model_name"],
                excluded_fields=[pipeline_config["entity_id"],],
                bigquery_source_input_uri="bq://" + str(export_data.outputs["input_table_paths"]),
                bigquery_destination_output_uri="bq://" + pipeline_config[
                    "batch_prediction_dest_data_table_uri"
                ],
                machine_type=pipeline_config["batch_prediction_machine_type"],
                encryption_spec_key_name=pipeline_config["encryption_spec_key_name"],
                labels=pipeline_config["labels"],
            )
            # TODO: update conduit-prediction-storage store batch prediction result to conduit-prediction-storage
            # Batch prediction with monitoring operation:
            batchpredict_with_monitor_op = batch_prediction_generator_monitor(
                project=pipeline_config["project"],
                location=pipeline_config["location"],
                job_display_name=pipeline_config["batch_prediction_job_display_name"],
                model=pipeline_config["model_name"],
                excluded_fields=[pipeline_config["entity_id"],],
                bigquery_source_input_uri="bq://" + str(export_data.outputs["input_table_paths"]),
                bigquery_destination_output_uri="bq://" + pipeline_config[
                    "batch_prediction_dest_data_table_uri"
                ]
                + "_drift_skew_monitor",
                train_data_uri=pipeline_config["train_data_table_uri"],
                target_column_name=pipeline_config["target_column_name"],
                monitor_output_uri=pipeline_config[
                    "model_batch_prediction_monitoring_stats_anomalies_base_directory"
                ],
                machine_type=pipeline_config["batch_prediction_machine_type"],
                encryption_spec_key_name=pipeline_config["encryption_spec_key_name"],
                user_emails=pipeline_config.get(
                    "model_batch_prediction_monitoring_user_emails", []
                ),  # Optional param - if None, config associated with this param won't be created
                default_skew_threshold=pipeline_config["model_batch_prediction_monitoring_default_skew_threshold"],
                skew_thresholds=pipeline_config.get(
                    "model_batch_prediction_monitoring_skew_thresholds", {}
                ),  # Optional param - if None, config associated with this param won't be created
                attribution_score_skew_thresholds=pipeline_config.get(
                    "model_batch_prediction_monitoring_attrib_skew_thresholds", {}
                ),  # Optional param - if None, config associated with this param won't be created
                drift_thresholds=pipeline_config.get(
                    "model_batch_prediction_monitoring_drift_thresholds", {}
                ),  # Optional param - if None, config associated with this param won't be created
                labels=pipeline_config["labels"],
            )
            monitor_explanation_result_op = monitor_explanation_result_writer(
                project=pipeline_config["project"],
                conduit_prediction_storage_table=pipeline_config[
                    "conduit_prediction_storage_table"
                ],
                skew_gcs_folder=batchpredict_with_monitor_op.outputs["skew_gcs_folder"],
                # TODO generate this programmatically in pipeline_config
                # or get it from the output of batch_prediction_generator_monitor
                monitor_explanation_dest_table=pipeline_config[
                    "batch_prediction_dest_data_table_uri"
                ]
                + "_drift_skew_monitor",
                encryption_spec_key_name=pipeline_config["encryption_spec_key_name"],
                labels=pipeline_config["labels"],
                entity_id=pipeline_config["entity_id"],
                execution_date=get_run_config.outputs["max_date"],
                model_name=pipeline_config["model_name"],
                dataset_name=pipeline_config["dataset_name"],
                enable_explanation=pipeline_config["enable_explanation"],
                enable_drift_monitoring=pipeline_config["enable_drift_monitoring"],
            )
            # TODO: update conduit storage after batch prediction result is updated
            #      example? update_monitor_result_op.after(batchpredict_result_update_op)
            monitor_explanation_result_op.after(batchpredict_op)
            exec_dt = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
            generate_predictions_temp = component(
                prediction_generator_temp,
                base_image="python:3.7.12",
                packages_to_install=[
                    "google-cloud-aiplatform",
                    "gcsfs",
                    "pandas",
                    "parquet",
                    "fastparquet",
                    "tqdm",
                    "db_dtypes",
                ],
            )(
                bigquery_source_input_uri=export_data.outputs["input_table_paths"],
                bigquery_destination_output_uri=pipeline_config[
                    "batch_prediction_dest_data_table_uri"
                ],
                entity_id=pipeline_config["entity_id"],
                endpoint_name=pipeline_config["model_endpoint_name"],
                model_name=pipeline_config["model_name"],
                project=pipeline_config["project"],
                location=pipeline_config["location"],
                encryption_spec_key_name=pipeline_config["encryption_spec_key_name"],
                feature_names=pipeline_config["feature_names"],
                labels=pipeline_config["labels"]
            )
            write_predictions = component(
                prediction_writer,
                base_image="us-central1-docker.pkg.dev/anbc-pdev/mleng-platform-docker-pdev/nk-test-protobuf:0.1",
                packages_to_install=[
                    "google-cloud-aiplatform",
                    "gcsfs",
                    "pandas",
                    "parquet",
                    "fastparquet",
                    "db_dtypes",
                    "protobuf",
                    "google-cloud-bigquery-storage"
                ],
            )(
                prediction_window_sec=pipeline_config["prediction_window_sec"],
                project=pipeline_config["project"],
                model_version_id=generate_predictions_temp.outputs["model_version_id"],
                model_name=pipeline_config["model_name"],
                model_id=generate_predictions_temp.outputs["model_id"],
                prediction_type=pipeline_config["prediction_type"],
                prediction_execution_date=get_run_config.outputs["max_date"],
                bq_output_table_path=generate_predictions_temp.outputs[
                    "bigquery_destination_output_uri"
                ],
                bq_prediction_storage_uri="anbc-pdev.mleng_platform_ent_pdev.conduit-prediction-storage",
                labels=pipeline_config["labels"],
                encryption_spec_key_name=pipeline_config["encryption_spec_key_name"],
            )

        # OOS Evaluation Pipeline
        with dsl.Condition(
            task_type == "oos_evaluation", name="oos-evaluation-pipeline"
        ):
            # Get values to create SQL
            get_run_config = component(
                run_config_generator_oos, base_image="python:3.7.12"
            )(
                schedule_frequency_day=pipeline_config["schedule_frequency_day"],
                base_date_str="2016-08-10", # NOTE: hard code base date for debugging. in production code, this argument should not be passed
            )
            # Generate input query
            generate_query = component(query_generator_oos, base_image="python:3.7.12")(
                input_query=pipeline_config["ground_truth_data_sql"],
                min_date=get_run_config.outputs["min_date"],
                max_date=get_run_config.outputs["max_date"],
                date_format=pipeline_config.get("date_format", ""),
            )
            TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
            # Mathc GT and Predictions
            match_prediction_ground_truth = component(
                prediction_ground_truth_matcher,
                base_image="python:3.7.12",
                packages_to_install=[
                    "google-cloud-bigquery",
                    "gcsfs",
                    "pandas",
                    "parquet",
                    "fastparquet",
                    "db_dtypes",
                ],
            )(
                ground_truth_query=generate_query.outputs["formatted_query"],
                ground_truth_min_date=get_run_config.outputs["min_date"],
                ground_truth_max_date=get_run_config.outputs["max_date"],
                project=pipeline_config["project"],
                entity_id=pipeline_config["entity_id"],
                gcs_output_dir="gs://mleng-platform-ent-data-pdev/nk/demo/sprint9/complete-monolithic-pipeline/oos-eval/{0}/".format(
                    TIMESTAMP
                ),
                model_name=pipeline_config["model_name"],
                labels=pipeline_config["labels"],
                bq_prediction_storage_uri="anbc-pdev.mleng_platform_ent_pdev.conduit-prediction-storage",
                encryption_spec_key_name=pipeline_config["encryption_spec_key_name"],
            )
            # Compute evaluation metrics
            compute_metrics = component(
                metric_computer,
                base_image="python:3.7.12",
                packages_to_install=[
                    "gcsfs",
                    "pandas",
                    "parquet",
                    "fastparquet",
                    "db_dtypes",
                    "scikit-learn",
                ],
            )(
                gt_pred_pair_path=match_prediction_ground_truth.outputs[
                    "gt_pred_pair_path"
                ],
                target_metrics=pipeline_config["eval_metrics"],
            )
            # Write OOS metrics to OOS metric storage
            write_oos_metrics_result_op = component(
                oos_metrics_result_writer,
                base_image="python:3.7.12",
                packages_to_install=["google-cloud-bigquery"],
            )(
                oos_metrics_result=compute_metrics.outputs["metrics_result"],
                # TODO: avoid hardcoding and import bq paths thorugh conduit package later
                bq_train_metric_storage_uri="anbc-pdev.mleng_platform_ent_pdev.conduit-train-storage",
                bq_oos_metric_storage_uri="anbc-pdev.mleng_platform_ent_pdev.conduit-oos-metric-storage",
                model_id=match_prediction_ground_truth.outputs["model_id"],
                model_version_id=match_prediction_ground_truth.outputs[
                    "model_version_id"
                ],
                model_name=pipeline_config["model_name"],
                project=pipeline_config["project"],
                labels=pipeline_config["labels"],
                encryption_spec_key_name=pipeline_config["encryption_spec_key_name"],
            )
            # Write ground truths into prediction storage
            backfill_ground_truth = component(
                ground_truth_backfiller,
                base_image="python:3.7.12",
                packages_to_install=[
                    "google-cloud-bigquery",
                    "gcsfs",
                    "pandas",
                    "parquet",
                    "fastparquet",
                    "db_dtypes",
                ]
            )(
                gt_pred_pair_path=match_prediction_ground_truth.outputs[
                    "gt_pred_pair_path"
                ],
                bq_dataset="mleng_platform_ent_pdev", # TODO: add this value in master input table
                labels=pipeline_config["labels"],
                bq_prediction_storage_uri="anbc-pdev.mleng_platform_ent_pdev.conduit-prediction-storage",
                project=pipeline_config["project"],
                model_name=pipeline_config["model_name"],
            )

    return pipeline


def compile_pipeline():
    PIPELINE_CONFIG_FILE_NAME = "new_pipeline_config.json"
    with open(PIPELINE_CONFIG_FILE_NAME, "r") as f:
        pipeline_config = json.load(f)

    pipeline = create_pipeline(pipeline_config)
    # compile pipeline
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=pipeline_config["pipeline_json_template_name"],
    )


if __name__ == "__main__":
    compile_pipeline()
