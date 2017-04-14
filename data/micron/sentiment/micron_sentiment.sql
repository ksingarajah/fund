SELECT v2tone, date FROM[gdelt-bq:gdeltv2.gkg] WHERE (organizations like '%micron technology%') AND documentidentifier like '%micron%' ;

SELECT v2tone, date
FROM [gdelt-bq:gdeltv2.gkg] 
WHERE (organizations like '%micron technology%')
AND (documentidentifier like '%micron%'); 