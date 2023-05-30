import rocks
import pandas as pd
import exploring_script as es


confirmed_sso = es.load_data(["ssnamenr"])
ssoname_sample = confirmed_sso["ssnamenr"].unique()
ast_identify = rocks.identify(ssoname_sample)

pdf_astid = pd.DataFrame(ast_identify, columns=["ast_name", "ast_number"])

print(pdf_astid)
pdf_astid.to_parquet("data/rocks_fink.parquet")
