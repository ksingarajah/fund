SELECT v2tone, date FROM[gdelt-bq:gdeltv2.gkg] WHERE (organizations like '%intel%') AND documentidentifier like '%intel%' ;

SELECT v2tone, date
FROM [gdelt-bq:gdeltv2.gkg] 
WHERE ((organizations like '%intel %' OR organizations like '%intel') AND organizations not like '%intel com%' AND organizations not like '%intel dem%' AND organizations not like '%flynn intel%' AND organizations not like '%justice department%' AND organizations not like '%national security agency%' AND organizations not like '%white house%' AND organizations not like '%intelligence%')
AND (documentidentifier like '%intel%');