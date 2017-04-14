SELECT v2tone, date FROM[gdelt-bq:gdeltv2.gkg] WHERE (organizations like '%amd inc%' OR v2organizations like '%advanced micro devices%');

SELECT v2tone, date
FROM [gdelt-bq:gdeltv2.gkg] 
WHERE (organizations like '% amd %' or organizations like '%advanced micro devices%')
AND (documentidentifier like '%amd%' or documentidentifier like '%advanced-micro-devices%'); 