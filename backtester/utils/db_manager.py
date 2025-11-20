"""Database management utilities for Optuna PostgreSQL backend.

This script provides functions to manage the PostgreSQL database and Optuna studies.
Usage:
    python -m backtester/utils/db_manager.py --help
"""

import argparse
import os

import optuna
import psycopg2  # type: ignore[import-untyped]
from dotenv import load_dotenv
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT  # type: ignore[import-untyped]

# Load environment variables
load_dotenv()


def get_db_connection(include_db: bool = True) -> dict[str, str]:
    """Get PostgreSQL connection parameters from environment variables."""
    db_user = os.getenv("POSTGRES_USER", "postgres")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "optuna_db")

    if not db_password:
        raise ValueError(
            "POSTGRES_PASSWORD environment variable is not set. Please check your .env file."
        )

    if include_db:
        return {
            "user": db_user,
            "password": db_password,
            "host": db_host,
            "port": db_port,
            "database": db_name,
        }
    else:
        return {"user": db_user, "password": db_password, "host": db_host, "port": db_port}


def get_storage_url() -> str:
    """Get Optuna storage URL."""
    params = get_db_connection(include_db=True)
    return f"postgresql://{params['user']}:{params['password']}@{params['host']}:{params['port']}/{params['database']}"


def create_database() -> None:
    """Create the Optuna database if it doesn't exist."""
    db_name = os.getenv("POSTGRES_DB", "optuna_db")

    try:
        # Connect to PostgreSQL server (not to a specific database)
        conn = psycopg2.connect(**get_db_connection(include_db=False))
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cursor.fetchone()

        if exists:
            print(f"✓ Database '{db_name}' already exists.")
        else:
            # Create database
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            print(f"✓ Database '{db_name}' created successfully.")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"✗ Error creating database: {e}")
        raise


def drop_database() -> None:
    """Drop the Optuna database."""
    db_name = os.getenv("POSTGRES_DB", "optuna_db")

    response = input(
        f"⚠️  Are you sure you want to DROP database '{db_name}'? This will delete ALL studies! (yes/no): "
    )
    if response.lower() != 'yes':
        print("Aborted.")
        return

    try:
        # Connect to PostgreSQL server (not to the database we're dropping)
        conn = psycopg2.connect(**get_db_connection(include_db=False))
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Terminate all connections to the database
        cursor.execute(
            f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{db_name}'
            AND pid <> pg_backend_pid()
        """
        )

        # Drop database
        cursor.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
        print(f"✓ Database '{db_name}' dropped successfully.")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"✗ Error dropping database: {e}")
        raise


def recreate_database() -> None:
    """Drop and recreate the database."""
    drop_database()
    create_database()
    print("✓ Database recreated successfully.")


def list_studies() -> None:
    """List all studies in the database."""
    try:
        storage_url = get_storage_url()
        studies = optuna.study.get_all_study_summaries(storage=storage_url)

        if not studies:
            print("No studies found in the database.")
            return

        print(f"\n{'Study Name':<30} {'Direction':<12} {'N Trials':<10} {'Best Value':<15}")
        print("-" * 70)

        for study_summary in studies:
            best_value = (
                f"{study_summary.best_trial.value:.4f}" if study_summary.best_trial else "N/A"
            )
            print(
                f"{study_summary.study_name:<30} {study_summary.direction.name:<12} {study_summary.n_trials:<10} {best_value:<15}"
            )

    except Exception as e:
        print(f"✗ Error listing studies: {e}")
        raise


def delete_study(study_name: str) -> None:
    """Delete a specific study."""
    response = input(f"⚠️  Are you sure you want to DELETE study '{study_name}'? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return

    try:
        storage_url = get_storage_url()
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print(f"✓ Study '{study_name}' deleted successfully.")

    except Exception as e:
        print(f"✗ Error deleting study: {e}")
        raise


def create_study(study_name: str, direction: str = "maximize") -> None:
    """Create a new study."""
    try:
        storage_url = get_storage_url()
        optuna.create_study(
            study_name=study_name, direction=direction, storage=storage_url, load_if_exists=False
        )
        print(f"✓ Study '{study_name}' created successfully with direction '{direction}'.")

    except optuna.exceptions.DuplicatedStudyError:
        print(
            f"✗ Study '{study_name}' already exists. Use a different name or delete the existing study first."
        )
    except Exception as e:
        print(f"✗ Error creating study: {e}")
        raise


def main() -> None:
    """Main function to manage PostgreSQL database and Optuna studies."""
    parser = argparse.ArgumentParser(
        description="Manage PostgreSQL database and Optuna studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtester/utils/db_manager.py --create-db
  python backtester/utils/db_manager.py --list-studies
  python backtester/utils/db_manager.py --delete-study portfolio_grid_search
  python backtester/utils/db_manager.py --create-study my_new_study --direction maximize
  python backtester/utils/db_manager.py --recreate-db
        """,
    )

    parser.add_argument("--create-db", action="store_true", help="Create the database")
    parser.add_argument("--drop-db", action="store_true", help="Drop the database (destructive!)")
    parser.add_argument(
        "--recreate-db", action="store_true", help="Drop and recreate the database (destructive!)"
    )
    parser.add_argument("--list-studies", action="store_true", help="List all studies")
    parser.add_argument("--delete-study", type=str, help="Delete a specific study by name")
    parser.add_argument("--create-study", type=str, help="Create a new study with the given name")
    parser.add_argument(
        "--direction",
        type=str,
        default="maximize",
        choices=["maximize", "minimize"],
        help="Direction for optimization (default: maximize)",
    )

    args = parser.parse_args()

    # Execute requested operation
    if args.create_db:
        create_database()
    elif args.drop_db:
        drop_database()
    elif args.recreate_db:
        recreate_database()
    elif args.list_studies:
        list_studies()
    elif args.delete_study:
        delete_study(args.delete_study)
    elif args.create_study:
        create_study(args.create_study, args.direction)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
