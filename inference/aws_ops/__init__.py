import boto3
import csv


class AWSOperations:
    def __init__(self, aws_region):
        self.aws_region = aws_region
        self.session = None

    def assume_role(self, role_arn, session_name):
        """Assume an IAM role and set up an AWS session with the temporary credentials."""
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

    def ensure_table_exists(self, table_name):
        """Ensure the DynamoDB table exists, creating it if necessary."""
        dynamodb = self.session.resource('dynamodb')
        existing_tables = dynamodb.meta.client.list_tables()['TableNames']

        if table_name in existing_tables:
            print(f"Table {table_name} already exists.")
            return

        print(f"Table {table_name} not found, creating...")
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {'AttributeName': 'feature', 'KeyType': 'HASH'}  # Primary key
            ],
            AttributeDefinitions=[
                {'AttributeName': 'feature', 'AttributeType': 'S'}  # String type
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        table.wait_until_exists()
        print(f"Table {table_name} created.")

    def upload_metadata_from_csv(self, csv_filepath, table_name):
        """Upload metadata from a CSV file to a DynamoDB table."""
        dynamodb = self.session.resource('dynamodb')
        table = dynamodb.Table(table_name)
        with open(csv_filepath, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                item = {k: v for k, v in row.items() if v}  # Exclude empty values
                table.put_item(Item=item)
                print(f"Uploaded item: {item}")
