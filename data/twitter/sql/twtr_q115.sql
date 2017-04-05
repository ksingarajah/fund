SELECT date, sourcecommonname, documentidentifier, v2tone, organizations, v2organizations FROM [gdelt-bq:gdeltv2.gkg] WHERE (organizations like '%twitter%') AND date> 20150414000000AND date<20150428000000 AND documentidentifier like '%twitter%' AND (themes like '%EPU_ECONOMY_HISTORIC%' OR themes like '%ECON_EARNINGSREPORT%');