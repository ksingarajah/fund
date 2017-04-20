SELECT v2tone, date
FROM [gdelt-bq:gdeltv2.gkg] 
WHERE (organizations like '%hp inc%' or organizations like '%hewlett-packard%')
AND (documentidentifier like '%hp%' or documentidentifier like '%hewlett-packard%'); 