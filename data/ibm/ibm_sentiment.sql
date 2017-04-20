SELECT v2tone, date FROM[gdelt-bq:gdeltv2.gkg] WHERE (organizations like '%ibm%') AND documentidentifier like '%ibm%' ;

SELECT v2tone, date
FROM [gdelt-bq:gdeltv2.gkg] 
WHERE (organizations like '%ibm%' or organizations like '%international business machines%')
AND (documentidentifier like '%ibm%' or documentidentifier like '%international-business-machines%'); 