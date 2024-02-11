#!/usr/bin/env python

from botocore.exceptions import ClientError
from env_config import EnvConfig
from aws_ops import AWSOperations


def main():
    config = EnvConfig()

    aws_ops = AWSOperations(config.aws_region)

    try:
        session_name = "OpportunityMetadataUploader"
        aws_ops.assume_role(config.metadata_uploader_role_arn, session_name)
        aws_ops.ensure_table_exists(config.opportunity_metadata_table_name)
        aws_ops.upload_metadata_from_csv(config.feature_metadata_path, config.opportunity_metadata_table_name)
    except ClientError as e:
        print(f"An error occurred: {e}")
        exit(1)

    print("Feature metadata upload complete.")


if __name__ == "__main__":
    main()
