SELECT date, sourcecommonname, documentidentifier, v2tone, organizations, v2organizations FROM [gdelt-bq:gdeltv2.gkg] WHERE (organizations like '%ibm%') AND date> 20160105000000AND date<20160119000000 AND documentidentifier like '%ibm%' AND (themes like '%EPU_ECONOMY_HISTORIC%' OR themes like '%ECON_EARNINGSREPORT%');