SELECT v2tone, date FROM[gdelt-bq:gdeltv2.gkg] WHERE (organizations like '%amazon%') AND documentidentifier like '%amazon%' ;

SELECT v2tone, date
FROM [gdelt-bq:gdeltv2.gkg] 
WHERE (organizations like '%amazon inc%' OR organizations like '%amazon web%')
AND documentidentifier like '%amazon%';