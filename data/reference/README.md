# Catálogo departamental

`departamentos.csv` es la fuente única de verdad para la unidad transversal del estudio.
Contiene los **24 departamentos oficiales del Perú**, con:

- `departamento_id`: código ISO 3166-2:PE;
- `departamento`: nombre oficial con tildes;
- `departamento_ascii`: alias usado por el panel y los archivos;
- `codigo_sisap`: código de consulta en SISAP;
- coordenadas, zona natural y marca de referencia Lima.

No forman parte del catálogo:

- `030201` Andahuaylas, provincia de Apurímac;
- `060401` Chota, provincia de Cajamarca;
- `060801` Jaén, provincia de Cajamarca;
- `070000` Callao, provincia constitucional.

Los HTML de esos códigos se conservan en `data/raw/_excluidos/` y no se leen en el pipeline.
