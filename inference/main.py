#!/usr/bin/env python
import uuid
from aws_ops import AWSOperations
from env_config import EnvConfig


def main():
    config = EnvConfig()

    aws_ops = AWSOperations(
        region=config.region,
        role_arn=config.role_arn,
        session_name=config.session_name + "_" + uuid.uuid4().hex,
        table_name=config.table_name,
        query_output_path=config.query_output_path,
        csv_paths=[
            config.products_csv_path,
            config.opportunity_csv_path,
            config.feature_metadata_path
        ]
    )

    if config.run_migrations:
        aws_ops.create_table()
        aws_ops.ingest_data_from_csv()

    aws_ops.get_products_by_opportunity(config.opportunity_id)


if __name__ == "__main__":
    main()
