import time

class DatabaseConnection:
    """
        Dynimically query snowflake
    """
    def __init__(self, env_name, dictkv):
        """
            env_name (str):
                python:
                    Connect Snowflake using sqlalchemy
                    SSO credential is stored as enrironment variables
                databricks:
                    Connect Snowflake using databricks utility
                    SSO credential is stored as secrets
            dictkv (dictionary):
                To retrieve SSO credentials, passed on to separate functions
        """
        self.env_name = env_name
        self.var_kv = dictkv
        self.options = None

    def _sf_python(self, var_dict):
        """
            var_dict (dict) example:
                {"user_var": "DB_SF_USER", "pass_var": "DB_SF_PASS"}
        """
        import os

        username = os.environ.get(var_dict.get("user_var"))
        password = os.environ.get(var_dict.get("pass_var"))
        self.options = dict(
            user=username,
            password=password,
            account="ted_tdc.us-east-1",
            database="prod_ac",
            schema=None,
            warehouse="reporting_01",
            role="PROD_AC_REVENUE_ANALYTICS_ADMIN",
        )

    def _sf_databricks(self, dbutils, var_dict):
        """
            var_dict (dict) example:
                {"scope": "chchen",
                 "user_var": "snowflakeuser",
                 "pass_var": "snowflakepassword"}
        """
        username = dbutils.secrets.get(
            scope=var_dict.get("scope"), key=var_dict.get("user_var")
        )
        password = dbutils.secrets.get(
            scope=var_dict.get("scope"), key=var_dict.get("pass_var")
        )
        self.options = dict(
            sfUrl="ted_tdc.us-east-1.snowflakecomputing.com/",
            sfUser=username,
            sfPassword=password,
            sfDatabase="prod_ac",
            sfSchema=None,
            sfWarehouse="reporting_01",
            sfRole="PROD_AC_REVENUE_ANALYTICS_ADMIN",
        )

    def return_query_df(self, sql_query):
        if self.env_name.lower() == "python":
            from sqlalchemy import create_engine
            import snowflake.connector as sf
            import pandas as pd

            self._sf_python(self.var_kv)
            engine = create_engine(
                "snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}&role={role}".format(
                    **self.options
                )
            )
            conn = engine.connect()
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            engine.dispose()
            return df
        elif self.env_name.lower() == "databricks":
            from pyspark.sql import SparkSession
            from pyspark.dbutils import DBUtils

            spark = SparkSession.builder.getOrCreate()
            try:
                dbutils = DBUtils(spark)
            except ImportError:
                import IPython

                dbutils = IPython.get_ipython().user_ns["dbutils"]
            dbutils.secrets.setToken("dkea7ece36c35981e852ad15b68b537bbbc3")
            self._sf_databricks(dbutils, self.var_kv)
            df = (
                spark.read.format("snowflake")
                .options(**self.options)
                .option("sfSchema", "DAAP_SLN_SCRATCH")
                .option("query", sql_query)
                .load()
            )
            return df
        else:
            print("Not a valid environment to execute query.")
        

def main(env_name):
    """
        Main code block for demo
    """
    sql_query = "SELECT seq4() FROM table(generator(rowcount=>(10)))"
    print(rf"Using Snowflake to query - {sql_query}")
    if env_name.lower() == "python":
        pass_var = {"user_var": "DB_SF_USER", "pass_var": "DB_SF_PASS"}
        my_sf = DatabaseConnection(env_name, pass_var)

        
        print("\nQuery from ", my_sf.env_name)
        test_df = my_sf.return_query_df(sql_query)
        print(type(test_df), test_df)
    elif env_name.lower() == "databricks":
        pass_var = {
            "scope": "chchen",
            "user_var": "snowflakeuser",
            "pass_var": "snowflakepassword",
        }
        my_sf = DatabaseConnection(env_name, pass_var)
        print("Query from ", my_sf.env_name)
        test_df = my_sf.return_query_df(sql_query)
        print(type(test_df), test_df.show())


if __name__ == "__main__":
    main("python")
    time.sleep(1)
    main("databricks")
    