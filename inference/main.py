#!/usr/bin/env python
import uuid
from aws_ops import AWSOperations
from env_config import EnvConfig


def main():
    config = EnvConfig()
    aws_ops = AWSOperations(config.region)

    aws_ops.assume_role(config.role_arn,
                        config.session_name + "_" + uuid.uuid4().hex)

    if config.run_migrations:
        aws_ops.create_table(config.table_name)
        aws_ops.upload_data_from_csv(
            [
                config.products_csv_path,
                config.opportunity_csv_path,
                config.feature_metadata_path
            ],
            config.table_name)

    aws_ops.query_table(table_name=config.table_name, opportunity_id=config.opportunity_id,
                        output_file=config.query_output_path)


if __name__ == "__main__":
    main()
