import boto3
import csv
import os
from loguru import logger


class AWSOperations:
    def __init__(self, region, role_arn, session_name, table_name, query_output_path, csv_paths):
        self.region = region
        self.role_arn = role_arn
        self.session_name = session_name
        self.table_name = table_name
        self.query_output_path = query_output_path
        self.csv_paths = csv_paths
        self.session = self.assume_role()

    def assume_role(self):
        sts_client = boto3.client('sts', region_name=self.region)
        assumed_role = sts_client.assume_role(
            RoleArn=self.role_arn,
            RoleSessionName=self.session_name
        )
        credentials = assumed_role['Credentials']
        return boto3.Session(
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

    def ingest_data_from_csv(self):
        dynamodb = self.session.resource('dynamodb')
        table = dynamodb.Table(self.table_name)

        for csv_path in self.csv_paths:
            file_type = os.path.basename(csv_path).split('.')[0]  # Assuming first part of the filename indicates type

            with open(csv_path, mode='r', encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    # Differentiate between file types to set PK and SK appropriately
                    if file_type == "products":
                        # For products, group by opportunity ID and use ASIN for uniqueness
                        pk_value = f"OPP#{row['opportunity']}"  # Group products under the same opportunity
                        sk_value = f"PROD#{row['asin']}"  # Unique identifier for each product
                        opportunity_value = row['opportunity']  # Set for GSI querying
                    elif file_type == "opportunities":
                        # For opportunities, use opportunity ID as PK, and a constant SK for all base details
                        pk_value = f"OPP#{row['opportunity']}"
                        sk_value = "DETAILS"  # A constant SK value for base opportunity details
                        opportunity_value = row['opportunity']  # Ensure this is set for consistency, might be redundant here
                    elif file_type == "features":
                        # For features, consider if they are globally unique or related to specific opportunities
                        pk_value = f"FEATURE#{row['feature']}"  # Unique identifier for each feature
                        sk_value = "METADATA"  # A constant SK value for feature metadata
                        # If features are related to opportunities, include opportunity ID in some form
                        # opportunity_value = f"OPP#{row['related_opportunity']}"  # Uncomment if applicable
                    else:
                        logger.warning(f"Unexpected file type: {file_type}")
                        continue

                    item = {
                        'PK': pk_value,
                        'SK': sk_value,
                        'opportunity': opportunity_value,  # This assumes all items can be related back to an opportunity
                        **{k: v for k, v in row.items() if v and k not in ['PK', 'SK', 'opportunity']}
                    }
                    try:
                        table.put_item(Item=item)
                        logger.debug(f"Uploaded item: {item}")
                    except Exception as e:
                        logger.error(f"Error uploading item {item}: {e}")

    def get_products_by_opportunity(self, opportunity_id):
        dynamodb = self.session.resource('dynamodb')
        table = dynamodb.Table(self.table_name)
        try:
            # Querying DynamoDB using the GSI
            response = table.query(
                IndexName='OpportunityIndex',
                KeyConditionExpression='opportunity = :opp',
                ExpressionAttributeValues={
                    ':opp': f'OPP#{opportunity_id}'
                },
                ProjectionExpression='asin, brand, category, price'
            )
            logger.debug(f"Query response: {response['Items']}")

            # Prepare data for CSV output
            products = response['Items']
            fieldnames = ['asin', 'brand', 'category', 'price']

            # Writing data to a CSV file
            with open(self.query_output_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for product in products:
                    writer.writerow({field: product.get(field, '') for field in fieldnames})
                logger.info(f"Product data for Opportunity ID {opportunity_id} written to {self.query_output_path}")
        except Exception as e:
            logger.error(f"Error fetching products for Opportunity ID {opportunity_id}: {e}")
