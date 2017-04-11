SELECT date, sourcecommonname, documentidentifier, v2tone, organizations, v2organizations FROM [gdelt-bq:gdeltv2.gkg] WHERE (organizations like '%apple inc%') AND date>20160112000000 AND date<20160126000000 AND documentidentifier like '%apple%' AND (themes like '%EPU_ECONOMY_HISTORIC%' OR themes like '%ECON_EARNINGSREPORT%');