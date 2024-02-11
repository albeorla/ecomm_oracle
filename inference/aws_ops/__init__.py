import boto3
import csv
import os
from boto3.dynamodb.conditions import Key
from loguru import logger


class AWSOperations:
    def __init__(self, region, role_arn, session_name, table_name, query_output_path):
        self.region = region
        self.role_arn = role_arn
        self.session_name = session_name
        self.table_name = table_name
        self.query_output_path = query_output_path
        self.session = None
        self.assume_role()

    def assume_role(self):
        sts_client = boto3.client('sts')
        assumed_role = sts_client.assume_role(
            RoleArn=self.role_arn,
            RoleSessionName=self.session_name
        )
        credentials = assumed_role['Credentials']
        self.session = boto3.Session(
            aws_access_key_id=credentials['AccessKeyId'],
            aws_secret_access_key=credentials['SecretAccessKey'],
            aws_session_token=credentials['SessionToken'],
            region_name=self.region
        )

    def create_table(self):
        dynamodb = self.session.resource('dynamodb')
        existing_tables = dynamodb.meta.client.list_tables()['TableNames']

        if self.table_name in existing_tables:
            logger.debug(f"Table {self.table_name} already exists.")
            dynamodb.Table(self.table_name).delete()
            logger.debug(f"Table {self.table_name} deleted.")
            dynamodb.meta.client.get_waiter('table_not_exists').wait(TableName=self.table_name)

        logger.debug(f"Creating table {self.table_name}...")
        table = dynamodb.create_table(
            TableName=self.table_name,
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

    def upload_data_from_csv(self, csv_filepaths):
        dynamodb = self.session.resource('dynamodb')
        table = dynamodb.Table(self.table_name)

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

    def get_products_by_opportunity(self, opportunity_id):
        # Replaces your query_table
        dynamodb = self.session.resource('dynamodb')
        table = dynamodb.Table(self.table_name)

        key_condition_expression = Key('opportunity').eq(':opp')
        expression_attribute_values = {
            ':opp': f'OPP#{opportunity_id}'
        }

        response = table.query(
            IndexName='OpportunityIndex',
            KeyConditionExpression=key_condition_expression,
            ExpressionAttributeValues=expression_attribute_values,
            FilterExpression=Key('SK').begins_with('PROD#')  # Filter for PRODUCTS only
        )

        with open(self.query_output_path, 'w') as file:
            file.write(str(response))
        logger.debug(f"Query response: {response}")
