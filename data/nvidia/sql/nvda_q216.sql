SELECT date, sourcecommonname, documentidentifier, v2tone, organizations, v2organizations FROM [gdelt-bq:gdeltv2.gkg] WHERE (organizations like '%nvidia%') AND date> 20150723000000AND date<20150806000000 AND documentidentifier like '%nvidia%' AND (themes like '%EPU_ECONOMY_HISTORIC%' OR themes like '%ECON_EARNINGSREPORT%');