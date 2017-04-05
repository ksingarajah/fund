SELECT date, sourcecommonname, documentidentifier, v2tone, organizations, v2organizations FROM [gdelt-bq:gdeltv2.gkg] WHERE (organizations like '%google%') AND date> 20160407000000AND date<20160421000000 AND documentidentifier like '%google%' AND (themes like '%EPU_ECONOMY_HISTORIC%' OR themes like '%ECON_EARNINGSREPORT%');