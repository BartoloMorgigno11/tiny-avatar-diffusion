# Four-model Streamlit UI

Launch the professor-facing UI after installing the demo extra:

```powershell
.venv\Scripts\python.exe -m streamlit run src/tiny_avatar_diffusion/final_streamlit_app.py
```

The selector exposes exactly `DDPM-U32`, `DDPM-C32`, `DDPM-U64`, and
`DDPM-C64`. Conditional models use the fixed four-attribute caption grammar;
unconditional models take no attributes. Every selected checkpoint is verified
against its SHA-256 before loading, and a missing file is never substituted.

The application is an inference demonstration, not a benchmark runner. The
1,000-step sampler can be slow on CPU.
