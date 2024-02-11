import boto3
import csv
import os
from boto3.dynamodb.conditions import Key
from loguru import logger


class AWSOperations:
    def __init__(self, aws_region):
        self.aws_region = aws_region
        self.session = None

    def assume_role(self, role_arn, session_name):
        sts_client = boto3.client('sts')
        assumed_role = sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName=session_name
        )
        credentials = assumed_role['Credentials']
        self.session = boto3.Session(
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken'],
            region_name=self.aws_region
        )

    def create_table(self, table_name):
        dynamodb = self.session.resource('dynamodb')
        existing_tables = dynamodb.meta.client.list_tables()['TableNames']

        if table_name in existing_tables:
            logger.debug(f"Table {table_name} already exists.")
            dynamodb.Table(table_name).delete()
            logger.debug(f"Table {table_name} deleted.")
            dynamodb.meta.client.get_waiter('table_not_exists').wait(TableName=table_name)

        logger.debug(f"Creating table {table_name}...")
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {'AttributeName': 'PK', 'KeyType': 'HASH'},  # Partition key
                {'AttributeName': 'SK', 'KeyType': 'RANGE'}  # Sort key
            ],
            AttributeDefinitions=[
                {'AttributeName': 'PK', 'AttributeType': 'S'},  # String type
                {'AttributeName': 'SK', 'AttributeType': 'S'},  # String type
                {'AttributeName': 'opportunity', 'AttributeType': 'S'}  # String type
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            },
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'OpportunityIndex',
                    'KeySchema': [
                        {'AttributeName': 'opportunity', 'KeyType': 'HASH'},  # GSI Partition key
                        {'AttributeName': 'SK', 'KeyType': 'RANGE'}  # GSI Sort key
                    ],
                    'Projection': {
                        'ProjectionType': 'ALL'
                    },
                    'ProvisionedThroughput': {
                        'ReadCapacityUnits': 5,
                        'WriteCapacityUnits': 5
                    }
                }
            ]
        )
        table.wait_until_exists()

    def upload_data_from_csv(self, csv_filepaths, table_name):
        dynamodb = self.session.resource('dynamodb')
        table = dynamodb.Table(table_name)

        for csv_filepath in csv_filepaths:
            file_type = os.path.basename(csv_filepath).split('_')[0]  # Identify file type based on name

            with open(csv_filepath, mode='r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    if file_type == "products":
                        # Example: PK = PROD#<ASIN>, SK = PRODUCT
                        pk_value = f"PROD#{row['asin']}"
                        sk_value = "PRODUCT"
                    elif file_type == "opportunity":
                        if 'asin' in row:
                            pk_value = f"OPP#{row['opportunity']}"
                            sk_value = f"SK#{row['asin']}"
                        else:
                            logger.warning(f"No 'asin' column found in the row: {row}")
                            continue
                    elif file_type == "feature":
                        pk_value = f"FEATURE#{row['feature']}"
                        sk_value = "METADATA"
                    else:
                        logger.warning(f"Unexpected file type: {file_type}")
                        continue

                    item = {
                        'PK': pk_value,
                        'SK': sk_value,
                        **{k: v for k, v in row.items() if v and k not in ['PK', 'SK']}
                    }
                    table.put_item(Item=item)
                    logger.debug(f"Uploaded item: {item}")

    def query_table(self, table_name, opportunity_id, output_file='data/gsi_query_sample.json'):
        dynamodb = self.session.resource('dynamodb')
        table = dynamodb.Table(table_name)

        key_condition_expression = Key('opportunity').eq(':opp') & Key('SK').begins_with('PROD#')
        expression_attribute_values = {
            ':opp': opportunity_id  # No need to prepend 'OPP#'
        }

        response = table.query(
            IndexName='OpportunityIndex',
            KeyConditionExpression=key_condition_expression,
            ExpressionAttributeValues=expression_attribute_values
        )

        with open(output_file, 'w') as file:
            file.write(str(response))
        logger.debug(f"Query response: {response}")

        # Return the items retrieved from the query
        return response['Items']
