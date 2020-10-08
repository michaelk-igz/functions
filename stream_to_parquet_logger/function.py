import os
import pandas as pd
import json
import datetime
from datetime import datetime


def init_context(context):
    setattr(context, 'batch', [])
    setattr(context, 'batch_size', int(os.getenv('BATCH_SIZE', 1024)))

    setattr(context, 'timestamp_key', os.getenv('TS_KEY'))
    setattr(context, 'timestamp_format', os.getenv('TS_FORMAT', '%Y-%m-%d %H:%M:%S.%f'))

    setattr(context, 'pq_partitions', ['pq_year', 'pq_month', 'pq_day', 'pq_hour'])

    setattr(context, 'target_path', os.getenv('TARGET_PATH'))
    os.makedirs(context.target_path, exist_ok=True)

    # in case of an inference stream set the names of features and predictions.
    features = os.getenv('FEATURES')
    if features is not None:
        features = features.split(',')
    setattr(context, 'features', features)

    predictions = os.getenv('PREDICTIONS')
    if predictions is not None:
        predictions = predictions.split(',')
    setattr(context, 'predictions', predictions)

    pass


def handler(context, event):
    if type(event.body) is dict:
        event_dict = event.body
    else:
        event_dict = json.loads(event.body)

    context.logger.info_with('Got invoked',
                             trigger_kind=event.trigger.kind,
                             event_body=event_dict)

    # for inference events
    if context.features is not None and context.predictions is not None:
        event_dict = flatten_inference_event(context, event_dict)

    event_with_time_partitions = add_time_partition_attributes(context, event_dict)

    # add the incoming event to the current batch
    context.batch.append(event_with_time_partitions)

    # check if batch size reached
    if context.batch_size == len(context.batch):
        written_records = write_batch(context)
        context.logger.info_with('Written batch',
                                 Writtent_records=written_records)
    pass


def flatten_inference_event(context, event):
    # add parsed features to the event
    feature_values = event['request']['instances'][0]
    event.update(zip(context.features, feature_values))

    # add parsed predictions to the event
    prediction_values = event['resp']
    event.update(zip(context.predictions, prediction_values))

    return event


def add_time_partition_attributes(context, event):
    if hasattr(context, 'timestamp_key') and event.get(context.timestamp_key) is not None:
        # parse the event time
        dt_object = datetime.strptime(event[context.timestamp_key], context.timestamp_format)
    else:
        # if event time is missing or not configured, use current datetime
        dt_object = datetime.now()

    # add the partition attributes
    event['pq_year'] = dt_object.strftime('%Y')
    event['pq_month'] = dt_object.strftime('%m')
    event['pq_day'] = dt_object.strftime('%d')
    event['pq_hour'] = dt_object.strftime('%H')

    return event


def write_batch(context):
    df = pd.DataFrame.from_records(context.batch)
    df.to_parquet(path=context.target_path, partition_cols=context.pq_partitions)
    # post write cleanup
    context.batch = []
    return len(df.index)
