using GreenFlux
using Documenter

makedocs(;
    modules=[GreenFlux],
    authors="Eduardo Cueto Mendoza",
    repo="https://github.com/EddieCueto/GreenFlux.jl/blob/{commit}{path}#L{line}",
    sitename="GreenFlux.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://EddieCueto.github.io/GreenFlux.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/EddieCueto/GreenFlux.jl",
)
