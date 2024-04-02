### A Pluto.jl notebook ###
# v0.19.39

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 433e3ada-ed59-11ee-0700-4514950dc68e
begin
	using Distributions
	using Random
	using Plots
	using PlutoUI
	using DataFrames
end

# ╔═╡ 90f95380-9cc9-429b-bd04-2585bae33e1e
md"#      Tarea 1"

# ╔═╡ 12bfb1d3-8b8b-402b-96d2-e82d7f52c507
md"#### Integrantes: 
* David Alejandro Alquichire Rincón
* Kevin Felipe Marroquín Olaya
* Tomas David Rodríguez Agudelo
"

# ╔═╡ 38e403bf-7531-4db3-9d40-19e1e63492ef
md"## Primer Punto"

# ╔═╡ d3bc992b-e6e2-40b4-9b42-1cf2914bdc14
md"Implemente el algoritmo (Gibbs sampler) usado en clase para generar muestras de una distribución que se aproxime a la distribución uniforme sobre las configuraciones factibles del modelo Hard-Core, en la rejilla cuadrada $k \times k \text{ }(3\leq k \leq 20)$.
Lo ideal es que se puedan visualizar las muestras y algunos pasos de la trayectoria de la cadena de Markov que condujo a la muestra. (Sugerencia: tome $X_{10000}$ o $X_{100000}$ como tiempo final)."

# ╔═╡ 1e96de02-f5fb-453d-98fe-eec48b9670b9
md"Primero definiremos la función `visualize` para graficar una configuración dada. Note que el argumento que se pasará, `configuration`, es una matriz de $k\times k$."

# ╔═╡ df6fb99c-dc23-4f37-b926-1a02bbf4d5fc
function visualize_conf(configuration)
	k = size(configuration, 1)
    plot(size=(k * 50, k * 50))
    
    plot!(legend=false, xaxis=false, yaxis=false, grid=false)
    hline!(1:k, color=:black, alpha=0.5, linewidth=0.5)
    vline!(1:k, color=:black, alpha=0.5, linewidth=0.5)

    for i in 1:k
		for j in 1:k
			if configuration[i,j] == 1
				scatter!([j], [k-i+1], color=:black, markersize=8)
			else
				scatter!([j], [k-i+1], color=:white, markersize=8, markerstrokecolor=:black)
			end
		end
	end
    
    current() 
end;

# ╔═╡ 27d9d2da-1995-4bc6-9ec3-e4229a6eb71b
md"A continuación, definimos la función `feasible` que verifica si un cambio propuesto en la configuración es factible según las reglas del modelo, tomando la configuración actual y las posiciones $(i, j)$ del vértice a cambiar, e iterando sobre los vecinos de dicha posición para verificar si alguno ya tiene valor $1$."

# ╔═╡ c9a5cea5-c208-4b31-a917-9a1b01389a1e
function feasible_conf(configuration, i, j)
	neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
	for (ni, nj) in neighbors
		if ni >= 1 && ni <= size(configuration, 1) && nj >= 1 && nj <= size(configuration, 2)
			if configuration[ni, nj] == 1
				return false
			end
		end
	end
	return true
end;

# ╔═╡ f8b1755c-dd19-4dbd-9bfe-06f54e299a50
md"Adicionalmente, definimos la función `valid_conf` para verificar si en general una configuración dada es factible."

# ╔═╡ 6a26e2ac-7705-4030-9dda-08e5bbf55a12
md"Ahora implementaremos el algoritmo de Gibb Sampler para el modelo hard-core mediante la función `hc_gibb`. 

El algoritmo es el siguiente: en cada tiempo entero $n+1$, hacemos lo siguiente:
1. Elegir un vértice $v\in V$ al azar (uniformemente).
2. Lanzar una moneda justa.
3. Si la moneda cae en cara y todos lo vecinos de $v$ toman el valor 0 en $X_n$, entonces $X_{n+1}(v)=1$; de lo contrario $X_{n+1}(v)=0$.
4. Para todos los vértices $w\in V$ ($w\neq v$), $X_{n+1}(w)=X_{n}(w)$."

# ╔═╡ df692ce9-6963-4c75-a771-14b220dd2d6b
function hc_gibb!(configuration)
	n = size(configuration, 1)
	i, j = rand(1:n), rand(1:n)
	if rand(Bernoulli(0.5)) == 1 # simular lanzamiento de moneda
		if feasible_conf(configuration, i, j)
			configuration[i, j] = 1
		end
	else
		configuration[i, j] = 0
	end
end;

# ╔═╡ 921a739f-afbf-4ee9-be44-796fa580b98f
md"Finalmente, definimos la función `precompute` para correr el Gibb Sampler y guardar las configuraciones generadas para posteriormente poder visualizar la cadena de Markov. Partiremos con una configuración inical aleatoria que puede ser factible o no, generada por la función `gen_config`. Note que podríamos partir de la configuración en la que todas las celdas están apagadas, lo cual es válido pues se probó que la cadena generada por este algoritmo para este modelo en particular es irreducible, aperiódica y reversible."

# ╔═╡ d39cb47f-b883-43ed-984b-0c11072f50a2
function gen_config(k)
	config = Array{Int}(undef, k, k)
	for i in 1:k
		for j in 1:k
			if rand(Bernoulli(0.5)) == 1
				config[i, j] = 1
			else
				config[i, j] = 0
			end
		end
	end
	config
end;

# ╔═╡ 5748c8a5-37c8-4faa-a91f-dec19237db4a
function precompute(k, total_steps, seed, z=nothing)
	Random.seed!(seed) # Para repetir el experimento con las mismas condiciones
    configurations = []
	flag = 0
    if(z != nothing)
		config = zeros(Int, k, k) # Configuración inicial vacia
	else
		config = gen_config(k) # Configuración inicial aleatoria
	end

    for step in 1:total_steps
        hc_gibb!(config)
		push!(configurations, copy(config)) # Guardar la configuración
    end

    configurations
end;

# ╔═╡ 90dda47b-2360-44f2-94fb-3f4758fdcf47
md"A continuación, establecemos los parámetros."

# ╔═╡ 7d595123-a942-47ac-85d0-abd7b437ea78
md"Tamaño de la grilla $k\times k$:"

# ╔═╡ 83dede68-1bbd-433e-8b9d-70c865b6a0d5
@bind k Slider(3:20, show_value=true, default=8)

# ╔═╡ 9f4fe13e-7287-43d5-9325-3a6d3e2b1402
function valid_conf(configuration)
	for i in 1:k
		for j in 1:k
			if (configuration[i, j] == 1)
				if feasible_conf(configuration, i, j) == false
					return false
				end
			end
		end
	end
	return true
end;

# ╔═╡ 095d3bd7-2cb9-409f-bb68-b68d541bfc6d
k

# ╔═╡ 74733451-fa7f-43cc-ab54-b0e2784c0713
md"Número de interaciones $steps$ (tiempo final):"

# ╔═╡ 57df806a-15bb-4ac4-9cad-cfc37e435867
begin
@bind steps Select([10000, 100000, 1000000])
end

# ╔═╡ 38434598-cf71-4c06-9dc6-bb46c2b8d996
steps

# ╔═╡ 7233979b-86c8-48a8-900a-3d10dd2ab03e
Markdown.parse("""
Generamos las configuraciones según los parámetros k=`$k` y steps=`$steps`.
""")

# ╔═╡ 09cfde5f-a644-4fa7-b7fa-5fb4beb394ba
configurations = precompute(k, steps, 73); # corremos la cadena

# ╔═╡ abcf1697-430b-48c8-8851-0acb6aab6c68
md"Obteniendo la siguiente visualización de las configuraciones obtenidas en cada tiempo $t$ ($1\leq t \leq steps$):

Nota: Si el valor de $t$ llega al valor de $steps$ reinicie el reloj. Para visualizar mediante un slider, edite la celda de abajo, descomente la línea 1 y comente la línea 2. "

# ╔═╡ 88e56f5e-e737-4845-b334-a36e344afb3e
@bind t Slider(1:steps, show_value=true)
#@bind t Clock(0.1, true)
#@bind t NumberField(1:steps)

# ╔═╡ 06f3008b-e221-443d-af27-db5e471cf777
t

# ╔═╡ cd721211-5e8c-482e-a241-a70d8ca5e1a6
visualize_conf(configurations[t])

# ╔═╡ 20b1ee19-0688-4a04-b097-78717935fd21
md"Salvaremos algunas de estas configuraciones para adjuntar en el repositorio."

# ╔═╡ b065f05b-dcc2-4bc1-be9d-90d4aa971dd3
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
for i in 1:3:50
	savefig(visualize_conf(configurations[i]), "./images/hard-core-configurations/X_"*string(i)*".png")
end
  ╠═╡ =#

# ╔═╡ 85a006b6-2e6f-4d78-8da9-758e5e727959
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
for i in 1:Int(steps/20):steps
	savefig(visualize_conf(configurations[i]), "./images/hard-core-configurations/X_"*string(i)*".png")
end
  ╠═╡ =#

# ╔═╡ b73ff478-6003-4c6f-b33c-c8facb30b2b2
md"En este punto, podemos analizar el tiempo mínimo en el que la cadena llega a una configuración factible partiendo desde una configuración inicial aleatoria. A continuación, generamos 10 cadenas disyuntas y anotamos dicho tiempo mínimo."

# ╔═╡ d8dcd884-59cf-4109-8f52-b48595892372
begin
iteraciones1 = 10
tiempo_minimo = DataFrame(Tiempo_mínimo = Int64[])
	
for i in 1:iteraciones1
	confs = precompute(k, steps, i) # Cada cadena generada se hace con una semilla distinta (i)
	num_particles1 = Int[]
	t_m = 0
	for config in confs
		t_m += 1
		if valid_conf(config)
			@goto found
		end
	end
	@label found
	push!(tiempo_minimo, [t_m])
end
	tiempo_minimo
end

# ╔═╡ 45498cc9-1bd0-4fe3-839d-eeeb53f7af00
function t_min(iteraciones)
	t_mins = Int[]
	t_max = 0
	for i in 1:iteraciones
		confs = precompute(k, 2000, i*37+iteraciones) # Cada cadena generada se hace con una semilla distinta (i*37+iteraciones)
		t_m = 0
		for config in confs
			t_m += 1
			if valid_conf(config)
				if (t_m > t_max) 
					t_max = t_m
				end
				push!(t_mins, t_m)
				@goto found
			end
		end
		@label found
	end
	return mean(t_mins), t_max
end;

# ╔═╡ b875075e-2834-4abe-9270-bacbefe3a63e
md"A continuación realizamos un análisis más profundo, generamos varias cadenas disyuntas, calculamos el promedio de los tiempos mínimos en los que cada cadena llega a una configuración factible y adicionalmente guardamos el máximo de tales tiempos mínimos."

# ╔═╡ 753c7328-290f-4616-a757-fd75c2154e8c
begin
	iterations_1 = [2^i for i in 3:15]
	t_proms = DataFrame(Cadenas_generadas = Int64[], Promedio_tiempos_mínimos = Float64[], Máximo_tiempos_mínimos = Int64[])
	for i in iterations_1
		prom_1, t_max = t_min(i)
		push!(t_proms, (Cadenas_generadas = i, Promedio_tiempos_mínimos = prom_1, Máximo_tiempos_mínimos = t_max))
	end
	t_proms
end

# ╔═╡ 72899a0b-3edd-4a6a-bc98-abf8b08560ff
md"Según el máximo de los tiempos mínimos, ¿es posible establecer una cota $c$ tal que, sin importar la configuración inicial, la cadena de Markov llegue a una configuración factible a lo sumo en el tiempo $c$?"

# ╔═╡ 9e0c101e-7ed2-46fc-979f-2dec6f3502cb
md"## Segundo Punto"

# ╔═╡ 7fbc57be-05cf-47d4-bf94-3e5acdefca54
md"Usar muestras generadas con lo hecho en el ejercicio anterior para estimar el número de partículas ''típico'' que tiene una configuración factible en la rejilla $k\times k$.
De hecho, lo ideal sería hacer un histograma. Verificar cómo cambia el histograma si en lo hecho en el primer punto se toman en vez de $X_{10000}$ o $X_{100000}$, otros tiempos de la cadena $\{X_t\}$."

# ╔═╡ 74da9482-2e9f-47ce-8ddc-e2fb6e9443ef
md"Por cada muestra factible generada en el primer ejercicio, contaremos el número de partículas y lo almacenaremos en el arreglo `num_particles` para posteriormente visualizar la frecuencia de dichas cantidades en un histograma."

# ╔═╡ eb6fbbba-2a73-491b-9186-26f0649fbd57
num_particles = Int[]; # Si se cambia el valor de k o de steps, correr esta línea de nuevo para actualizar el histograma

# ╔═╡ aae9c289-db4b-4ce1-a3fa-a314805b579e
for config in configurations
	flag = 0
	if (flag == 1 || valid_conf(config))
		flag = 1
		num = sum(config)
		push!(num_particles, num)
	end
end

# ╔═╡ 7edee3e1-a2d8-4c94-a49c-41d16fcefb30
md"Obteniendo el siguiente histograma:"

# ╔═╡ c168baa0-8768-4357-8efe-1d2d8c4c64e1
begin
cant = round(mean(num_particles), RoundDown)
por = round(100*cant/(k*k), sigdigits=4)
h1 = histogram(num_particles, bins=maximum(num_particles)-minimum(num_particles), label="",
          xlabel="Número de partículas", ylabel="Frecuencia", 
          title="Número de partículas en una grilla $k x $k")
vline!([mean(num_particles)], label="Promedio", color=:red)
end

# ╔═╡ 46487daa-4f3d-43e5-9898-83df4149fbf8
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
savefig(h1, "./images/histograms/"*string(k)*"x"*string(k)*"_"*string(steps)*".png");
  ╠═╡ =#

# ╔═╡ 3720c160-9f59-47da-ad69-bb09860ff9c3
Markdown.parse("""
Obteniendo una cantidad de partículas promedio de `$cant` (redondeando hacia abajo). Es decir, el ≈`$por`% del número total de celdas disponibles en la grilla (`$k`x`$k`).
""")

# ╔═╡ 2d839d89-e07a-4646-89df-a13fb4861a14
md"Ahora, consideraremos muestras ajenas a las del primer ejercicio:"

# ╔═╡ 893e4f03-94be-4ac4-91e4-a682d3356f7e
md"Tomando $10000$ como tiempo final de la cadena, obtenemos el siguiente histograma:"

# ╔═╡ 260a909f-4cbf-4dea-9683-0b2e191ebea0
begin
confs_10k = precompute(k, 10000, 43) # semilla = 43
par_nums_10k = Int[]
flag1 = 0
for conf in confs_10k
	if (flag1 == 1 || valid_conf(conf))
		flag1 = 1
		push!(par_nums_10k, sum(conf))
	end
end
prom_part_10k = mean(par_nums_10k)
	
cant1 = round(prom_part_10k, RoundDown)
por1 = round(100*cant1/(k*k), sigdigits=4)
	
histogram(par_nums_10k, bins=maximum(par_nums_10k)-minimum(par_nums_10k), label="",
          xlabel="Número de partículas", ylabel="Frecuencia", 
          title="Número de partículas en una grilla $k x $k")
vline!([prom_part_10k], label="Promedio", color=:red)
end

# ╔═╡ 62929933-3388-4531-876b-48ded1dc0fcf
Markdown.parse("""
Obteniendo una cantidad de partículas promedio de `$cant1` (redondeando hacia abajo). Es decir, el ≈`$por1`% del número total de celdas disponibles en la grilla (`$k`x`$k`).
""")

# ╔═╡ db9850bf-da1e-4d7c-9d79-519acbfc0b5d
md"Tomando $100000$ como tiempo final de la cadena, obtenemos el siguiente histograma:"

# ╔═╡ 60fb3013-832e-426f-ae65-ebbb09e68dc3
begin
confs_100k = precompute(k, 100000, 123) # semilla = 123
par_nums_100k = Int[]
flag = 0
for conf in confs_100k
	if (flag == 1 || valid_conf(conf))
		flag = 1
		push!(par_nums_100k, sum(conf))
	end
end
prom_part_100k = mean(par_nums_100k)

cant2 = round(prom_part_100k, RoundDown)
por2 = round(100*cant2/(k*k), sigdigits=4)

	
h2 = histogram(par_nums_100k, bins=maximum(par_nums_100k)-minimum(par_nums_100k), label="",
          xlabel="Número de partículas", ylabel="Frecuencia", 
          title="Número de partículas en una grilla $k x $k")
vline!([prom_part_100k], label="Promedio", color=:red)
end

# ╔═╡ f9eb2fc0-574b-4d26-aa8a-04f60abca472
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
savefig(h2, "./images/histograms/"*string(k)*"x"*string(k)*"_"*string(100000)*".png");
  ╠═╡ =#

# ╔═╡ 88b5f3b4-d22c-4782-968a-526eb6138437
Markdown.parse("""
Obteniendo una cantidad de partículas promedio de `$cant2` (redondeando hacia abajo). Es decir, el ≈`$por2`% del número total de celdas disponibles en la grilla (`$k`x`$k`).
""")

# ╔═╡ 5e28e419-e1ef-4b58-9109-5a653f8cb62f
md"A continuación, consideraremos tiempos finales (número de iteraciones) distintos a $10000$ y $100000$; y mostraremos los resultados en una tabla. Cada tiempo final considerado se realizará con muestras distintas (cadenas disyuntas). En este caso, cada cadena comenzará con una configuración inicial en la que todas las celdas están vacías."

# ╔═╡ 5d6a7423-f31a-4443-b6a1-333e12bf84c3
begin
iteraciones = [2^i for i in 3:20] 
df = DataFrame(Iteraciones = Int[], Promedio_partículas = Int64[], Porcentaje_partículas = Float64[])

porcentajes = Float64[]
cantidades = Int[]
	
for iters in iteraciones
	confs = precompute(k, iters, iters, 1) # semilla = iters
	par_nums = Int[]
	flag = 0
	for conf in confs
		if (flag == 1 || valid_conf(conf))
			flag = 1
			push!(par_nums, sum(conf))
		end
	end
	prom_part = floor(mean(par_nums))
	perc_part = round((prom_part/(k*k)) * 100, sigdigits=4)
	push!(df, (Iteraciones = iters, Promedio_partículas = prom_part, Porcentaje_partículas = perc_part))
	push!(porcentajes, perc_part)
	push!(cantidades, prom_part)
end
	prom_por = round(last(porcentajes), sigdigits=2);
end

# ╔═╡ 25c629ef-dd48-4af0-91c8-158ab1fbf7f0
md"Obteniendo:"

# ╔═╡ dc8a4b0d-b6f4-4341-a1e6-6a66a0777544
df

# ╔═╡ dc62d133-62be-4ef8-a0a1-841042ba3afb
Markdown.parse("""
Concluimos que el porcentaje de partículas presente en la grilla oscilará al rededor del `$prom_por`% (con respecto al número total de celdas `$k` x `$k`) a medida que el número de iteraciones aumenta.
""")

# ╔═╡ d8ea92ca-6281-4b90-b54a-1c479cb0f5f2
md"## Tercer Punto"

# ╔═╡ 42c58e86-9449-4997-9a93-61870cb9d251
md"Replicar lo hecho en el primer punto para $q-$coloraciones. ($2\leq q\leq 10, 3\leq k \leq 20$)."

# ╔═╡ 863f2c96-32ca-435a-b0d6-9f3b6e566870
md"Modificaremos sutilmente las funciones del primer ejercicio:"

# ╔═╡ 641dd88b-23a1-4f16-a492-19dbcc256e8b
md"Para visualizar una $q$-coloración:"

# ╔═╡ 928d6171-3396-4d26-8941-53027123b873
function visualize_coloration(coloration, q)
	colors = Plots.palette(:viridis, q)
	k = size(coloration, 1)
    plot(size=(k * 50, k * 50))
    
    plot!(legend=false, xaxis=false, yaxis=false, grid=false)
    hline!(1:k, color=:black, alpha=0.5, linewidth=0.5)
    vline!(1:k, color=:black, alpha=0.5, linewidth=0.5)

    for i in 1:k
		for j in 1:k
			scatter!([j], [k-i+1], color=colors[coloration[i, j]+1], markersize=8)
		end
	end
    
    current() 
end;

# ╔═╡ f630922d-e981-4948-9aaf-83371a6b4b90
md"Para generar una coloración aleatoria:"

# ╔═╡ 12f472c3-9126-4877-802f-ab68f103f3b1
function gen_coloration(k, q)
	config = Array{Int}(undef, k, k)
	for i in 1:k
		for j in 1:k
			config[i, j] = rand(0:q-1) # color aleatorio [0, 1, ..., q-1]
		end
	end
	config
end;

# ╔═╡ 99fdfb03-1af4-41e2-83b9-4bebeca74e96
md"Para verificar si se puede o no colorear el vértice (`i`,`j`) con el color `color`."

# ╔═╡ de296574-c8ea-400a-9d94-da78094570ea
function feasible_coloration(coloration, i, j, color)
	neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
	for (ni, nj) in neighbors
		if ni >= 1 && ni <= size(coloration, 1) && nj >= 1 && nj <= size(coloration, 2)
			if coloration[ni, nj] == color
				return false
			end
		end
	end
	return true
end;

# ╔═╡ 33df57e7-e246-42ea-b909-fddd09e85b34
md"Para verificar si una coloración es q-coloración:"

# ╔═╡ 54d9d3b3-4bc0-41bb-b838-574ee48848ba
function valid_coloration(coloration, k)
	for i in 1:k
		for j in 1:k
			if feasible_coloration(coloration, i, j, coloration[i, j]) == false
				return false
			end
		end
	end
	return true
end;

# ╔═╡ ccf0ffc9-9963-45d6-be4d-abb7f8360b26
md"Implementación del Gibb Sampler para el caso de $q$-coloración:"

# ╔═╡ 4bfa2623-b78b-4154-a6a2-1d29b6f0facc
function col_gibbs!(coloration, q)
	n = size(coloration, 1)
	i, j = rand(1:n), rand(1:n)
	color = rand(0:q-1)
	if feasible_coloration(coloration, i, j, color)
		coloration[i, j] = color
	end
end;

# ╔═╡ e64e54a6-88a9-4261-adb6-307954118ed4
md"Para correr la cadena:"

# ╔═╡ 4d77250e-731f-45a8-8857-8e1e009d603b
function precompute_colorations(k, q, total_steps, seed, z = nothing)
	Random.seed!(seed) # Para repetir el experimento con las mismas condiciones
    configurations = []
	if (z != nothing)
		config = zeros(Int, k, k)
	else
		config = gen_coloration(k, q)
	end

    for step in 1:total_steps
        col_gibbs!(config, q)
        push!(configurations, copy(config)) # Guardar la configuración
    end

    configurations
end;

# ╔═╡ 225d07c8-ebab-403a-aa50-00bb42ea3522
md"A continuación establecemos los parámetros para la $q$-coloración en una grilla de tamaño $k\times k$. Primero consideraremos $4 \leq q \leq 10$."

# ╔═╡ 07e23d0b-2d12-4cff-9e7b-320216b246ee
@bind k_ Slider(3:20, show_value=true)

# ╔═╡ 4552e0b9-8fba-4f33-a33d-9c50572a7d10
k_

# ╔═╡ 69497b83-da6a-455a-a5ed-1e6d828722aa
@bind q_ Slider(4:10, show_value=true)

# ╔═╡ ef33ae25-bf5b-45a8-b31c-e70c83952345
q_

# ╔═╡ d913bbcd-7baf-43c6-8153-4d71777951ce
@bind steps_ Select([10000, 100000, 1000000])

# ╔═╡ bab2f964-0b54-4a51-ba1e-99ce1c6ec9c4
steps_

# ╔═╡ 87195d68-50b6-4346-8183-c79aedec9b88
md"Corremos la cadena"

# ╔═╡ afe83872-dc58-4a70-b4dd-d381b37f7e5f
colorations = precompute_colorations(k_, q_, steps_, 37);

# ╔═╡ 16b223bd-9dfe-4d69-b2d3-6ce03eca5120
md"Obteniendo la siguiente visualización:"

# ╔═╡ 22170b40-05f8-49ae-9386-0268fc2d192f
@bind t_ Slider(1:steps_, show_value=true)
#@bind t_ Clock(0.1, true)
#@bind t_ NumberField(1:steps_)

# ╔═╡ 20d6e8f9-9b70-49b6-9081-dc509f506795
visualize_coloration(colorations[t_], q_)

# ╔═╡ f9688f68-d6cd-4a18-aab9-3beb80848c4f
md"Verifiquemos que se llega a una $q$-coloración:"

# ╔═╡ b773915d-7bed-46eb-9277-47128a346b99
valid_coloration(last(colorations), k_)

# ╔═╡ d895b7ac-67d7-424a-b577-de74fbd9a735
Markdown.parse("""
Encontremos el tiempo mínimo en el que se obtiene una $q_-coloración:
""")

# ╔═╡ ae083570-8d89-41ee-a2a0-1a73cea63037
function num_valid(colorations, k)
	for i in 1:steps_
		if valid_coloration(colorations[i], k)
			return i
		end
	end
end;

# ╔═╡ a1ae3b64-e4aa-4835-a1a4-3b14a4a1c9c7
num_valid(colorations, k_)

# ╔═╡ a13aa1b2-c2e2-4ce8-aaad-6a18346c34a8
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
savefig(visualize_coloration(colorations[8922], q_), "./images/colorations/"*string(k_)*"x"*string(k_)*"_"*string(q_)*"_8922.png");
  ╠═╡ =#

# ╔═╡ f601491c-da58-4e7e-b7ab-9a9ca109aae7
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
savefig(visualize_coloration(colorations[8923], q_), "./images/colorations/"*string(k_)*"x"*string(k_)*"_"*string(q_)*"_8923.png");
  ╠═╡ =#

# ╔═╡ b154e07a-2bd4-426a-ab10-2da986304503
md"A continuación, generaremos cadenas disyuntas para $3\leq k \leq 20$ y anotaremos el tiempo mínimo en el que cada cadena llega a una configuración factible."

# ╔═╡ 488b21ad-c032-4bb6-9f13-dca382134254
begin
df3 = DataFrame(k=Int64[], Tiempo_minimo=Int64[])
for size in 3:20
	cols = precompute_colorations(size, q_, 15000, size)
	ts = num_valid(cols, size)
	push!(df3, (size, ts))
end
df3
end

# ╔═╡ 8871b0e3-0a02-4516-80c8-fb741d730eb0
Markdown.parse("""A priori, este valor depende de la coloración inicial, note que en este caso la cadena generada para k=`$k_` llega a una q-coloración en un tiempo distinto al de la cadena generada en el ejemplo inicial (también de k=`$k_`)""")

# ╔═╡ 6aab0ee0-d6bf-47f6-abbf-870bab6e2385
md"Los casos $q=2$ y $q=3$ son especiales pues, dependiendo de la configuración inicial es posible llegar a una clase recurrente cuyas coloraciones no son una $q$-coloración. Veamos un ejemplo:"

# ╔═╡ 31e405ce-1eca-4458-9615-d142cf872e38
cols_3 = precompute_colorations(8, 3, 100000, 301); # Semilla=301

# ╔═╡ fa80e92c-123b-4338-848e-e504897d64cf
@bind t_3 Slider(1:100000, show_value=true)

# ╔═╡ 45e59e94-e174-4112-98e3-84e30701bec4
t_3

# ╔═╡ 659996e3-3f72-4cf9-89a0-004285cc321e
visualize_coloration(cols_3[t_3], 3)

# ╔═╡ 21e97b17-8e7b-4f7d-8ad6-9b3f56f0c57a
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
savefig(visualize_coloration(cols_3[731], 3), "./images/colorations/"*string(k_)*"x"*string(k_)*"_not"*string(3)*".png");
  ╠═╡ =#

# ╔═╡ 36651fa6-8bb1-4eca-8de8-93fe8d55eff0
md"Veamos que no se llega a una 3-coloración:"

# ╔═╡ 52dff1a0-7f79-4188-aaa6-25a563d40748
valid_coloration(last(cols_3), 8)

# ╔═╡ 71b13205-5459-47c7-bf1f-30b1ca2fdac3
md"Observe que no hay forma de pasar de la clase a la que se llegó a una $q$-coloración, pues muchos de los vértices quedaron determinados a estar con el correspondiente color."

# ╔═╡ d4f6f713-a684-4dcb-80b0-2021b3d5902d
md"Veamos un ejemplo en el que se llega a una $3-coloración válida."

# ╔═╡ 67354b57-ad30-4791-942e-3ed62e9e3571
cols_3_ = precompute_colorations(8, 3, 100000, 129); # Semilla=129

# ╔═╡ 38941b87-1e84-4345-96fa-3bc1eb9b017b
@bind t_3_ Slider(1:100000, show_value=true)

# ╔═╡ 271584dc-9617-44d2-9993-84c4aeab11c9
t_3_

# ╔═╡ 47605488-5844-493c-9a69-f4e4bffa4a57
visualize_coloration(cols_3_[t_3_], 3)

# ╔═╡ 915b2a97-377b-415b-a12a-dcd62167ac12
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
savefig(visualize_coloration(cols_3[731], 3), "./images/colorations/"*string(k_)*"x"*string(k_)*"_"*string(3)*".png");
  ╠═╡ =#

# ╔═╡ c48d0ca0-f847-4fc5-9e65-7d2d82cb4e47
md"Veamos que es una $3$-coloración:"

# ╔═╡ a75397ea-94df-44c7-b65e-3cc2a3b499bd
valid_coloration(last(cols_3_), 8)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
DataFrames = "~1.6.1"
Distributions = "~0.25.107"
Plots = "~1.40.2"
PlutoUI = "~0.7.58"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.1"
manifest_format = "2.0"
project_hash = "46810ec7b7d3113e13d9265e99b8ec0ab3924060"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0f748c81756f2e5e6854298f11ad8b2dfae6911a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a4c43f59baa34011e303e76f5c8c91bf58415aaf"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.0+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "6cbbd4d241d7e6579ab354737f4dd95ca43946e1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "0f4b5d62a88d8f59003e43c25a8a90de9eb76317"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.18"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "3437ade7073682993e092ca570ad68a2aba26983"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.3"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a96d5c713e6aa28c242b0d25c1347e258d6541ab"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.3+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "359a1ba2e320790ddbe4ee8b4d54a305c0ea2aff"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.0+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "8e59b47b9dc525b70550ca082ce85bcd7f5477cd"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.5"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3336abae9a713d2210bb57ab484b1e065edd7d23"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cad560042a7cc108f5a4c24ea1431a9221f22c1b"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.2"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dae976433497a2f841baadea93d27e68f1a12a97"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.39.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0a04a1318df1bf510beb2562cf90fb0c386f58c4"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.39.3+1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "af81a32750ebc831ee28bdaaba6e1067decef51e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.2"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60e3045590bd104a16fefb12836c00c0ef8c7f8c"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "7b1a9df27f072ac4c9c7cbe5efb198489258d1f5"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "3c403c6590dd93b36752634115e20137e79ab4df"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.2"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "88b895d13d53b5577fd53379d913b9ab9ac82660"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "cef0472124fab0695b58ca35a77c6fb942fdab8a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "71509f04d045ec714c4748c785a59045c3736349"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.7"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "3c793be6df9dd77a0cf49d80984ef9ff996948fa"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "07e470dabc5a6a4254ffebc29a1b3fc01464e105"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "31c421e5516a6248dfb22c194519e37effbf1f30"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d7015d2e18a5fd9a4f47de711837e980519781a4"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.43+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─90f95380-9cc9-429b-bd04-2585bae33e1e
# ╟─12bfb1d3-8b8b-402b-96d2-e82d7f52c507
# ╟─38e403bf-7531-4db3-9d40-19e1e63492ef
# ╟─d3bc992b-e6e2-40b4-9b42-1cf2914bdc14
# ╠═433e3ada-ed59-11ee-0700-4514950dc68e
# ╟─1e96de02-f5fb-453d-98fe-eec48b9670b9
# ╠═df6fb99c-dc23-4f37-b926-1a02bbf4d5fc
# ╟─27d9d2da-1995-4bc6-9ec3-e4229a6eb71b
# ╠═c9a5cea5-c208-4b31-a917-9a1b01389a1e
# ╟─f8b1755c-dd19-4dbd-9bfe-06f54e299a50
# ╠═9f4fe13e-7287-43d5-9325-3a6d3e2b1402
# ╟─6a26e2ac-7705-4030-9dda-08e5bbf55a12
# ╠═df692ce9-6963-4c75-a771-14b220dd2d6b
# ╟─921a739f-afbf-4ee9-be44-796fa580b98f
# ╠═d39cb47f-b883-43ed-984b-0c11072f50a2
# ╠═5748c8a5-37c8-4faa-a91f-dec19237db4a
# ╟─90dda47b-2360-44f2-94fb-3f4758fdcf47
# ╟─7d595123-a942-47ac-85d0-abd7b437ea78
# ╟─83dede68-1bbd-433e-8b9d-70c865b6a0d5
# ╠═095d3bd7-2cb9-409f-bb68-b68d541bfc6d
# ╟─74733451-fa7f-43cc-ab54-b0e2784c0713
# ╟─57df806a-15bb-4ac4-9cad-cfc37e435867
# ╠═38434598-cf71-4c06-9dc6-bb46c2b8d996
# ╟─7233979b-86c8-48a8-900a-3d10dd2ab03e
# ╠═09cfde5f-a644-4fa7-b7fa-5fb4beb394ba
# ╟─abcf1697-430b-48c8-8851-0acb6aab6c68
# ╠═88e56f5e-e737-4845-b334-a36e344afb3e
# ╠═06f3008b-e221-443d-af27-db5e471cf777
# ╠═cd721211-5e8c-482e-a241-a70d8ca5e1a6
# ╟─20b1ee19-0688-4a04-b097-78717935fd21
# ╠═b065f05b-dcc2-4bc1-be9d-90d4aa971dd3
# ╠═85a006b6-2e6f-4d78-8da9-758e5e727959
# ╟─b73ff478-6003-4c6f-b33c-c8facb30b2b2
# ╠═d8dcd884-59cf-4109-8f52-b48595892372
# ╟─45498cc9-1bd0-4fe3-839d-eeeb53f7af00
# ╟─b875075e-2834-4abe-9270-bacbefe3a63e
# ╟─753c7328-290f-4616-a757-fd75c2154e8c
# ╟─72899a0b-3edd-4a6a-bc98-abf8b08560ff
# ╟─9e0c101e-7ed2-46fc-979f-2dec6f3502cb
# ╟─7fbc57be-05cf-47d4-bf94-3e5acdefca54
# ╟─74da9482-2e9f-47ce-8ddc-e2fb6e9443ef
# ╠═eb6fbbba-2a73-491b-9186-26f0649fbd57
# ╠═aae9c289-db4b-4ce1-a3fa-a314805b579e
# ╟─7edee3e1-a2d8-4c94-a49c-41d16fcefb30
# ╟─c168baa0-8768-4357-8efe-1d2d8c4c64e1
# ╠═46487daa-4f3d-43e5-9898-83df4149fbf8
# ╟─3720c160-9f59-47da-ad69-bb09860ff9c3
# ╟─2d839d89-e07a-4646-89df-a13fb4861a14
# ╟─893e4f03-94be-4ac4-91e4-a682d3356f7e
# ╟─260a909f-4cbf-4dea-9683-0b2e191ebea0
# ╟─62929933-3388-4531-876b-48ded1dc0fcf
# ╟─db9850bf-da1e-4d7c-9d79-519acbfc0b5d
# ╟─60fb3013-832e-426f-ae65-ebbb09e68dc3
# ╠═f9eb2fc0-574b-4d26-aa8a-04f60abca472
# ╟─88b5f3b4-d22c-4782-968a-526eb6138437
# ╟─5e28e419-e1ef-4b58-9109-5a653f8cb62f
# ╟─5d6a7423-f31a-4443-b6a1-333e12bf84c3
# ╟─25c629ef-dd48-4af0-91c8-158ab1fbf7f0
# ╟─dc8a4b0d-b6f4-4341-a1e6-6a66a0777544
# ╟─dc62d133-62be-4ef8-a0a1-841042ba3afb
# ╟─d8ea92ca-6281-4b90-b54a-1c479cb0f5f2
# ╟─42c58e86-9449-4997-9a93-61870cb9d251
# ╟─863f2c96-32ca-435a-b0d6-9f3b6e566870
# ╟─641dd88b-23a1-4f16-a492-19dbcc256e8b
# ╠═928d6171-3396-4d26-8941-53027123b873
# ╟─f630922d-e981-4948-9aaf-83371a6b4b90
# ╠═12f472c3-9126-4877-802f-ab68f103f3b1
# ╟─99fdfb03-1af4-41e2-83b9-4bebeca74e96
# ╠═de296574-c8ea-400a-9d94-da78094570ea
# ╟─33df57e7-e246-42ea-b909-fddd09e85b34
# ╠═54d9d3b3-4bc0-41bb-b838-574ee48848ba
# ╟─ccf0ffc9-9963-45d6-be4d-abb7f8360b26
# ╠═4bfa2623-b78b-4154-a6a2-1d29b6f0facc
# ╟─e64e54a6-88a9-4261-adb6-307954118ed4
# ╠═4d77250e-731f-45a8-8857-8e1e009d603b
# ╟─225d07c8-ebab-403a-aa50-00bb42ea3522
# ╟─07e23d0b-2d12-4cff-9e7b-320216b246ee
# ╠═4552e0b9-8fba-4f33-a33d-9c50572a7d10
# ╟─69497b83-da6a-455a-a5ed-1e6d828722aa
# ╠═ef33ae25-bf5b-45a8-b31c-e70c83952345
# ╟─d913bbcd-7baf-43c6-8153-4d71777951ce
# ╠═bab2f964-0b54-4a51-ba1e-99ce1c6ec9c4
# ╟─87195d68-50b6-4346-8183-c79aedec9b88
# ╠═afe83872-dc58-4a70-b4dd-d381b37f7e5f
# ╟─16b223bd-9dfe-4d69-b2d3-6ce03eca5120
# ╠═22170b40-05f8-49ae-9386-0268fc2d192f
# ╠═20d6e8f9-9b70-49b6-9081-dc509f506795
# ╟─f9688f68-d6cd-4a18-aab9-3beb80848c4f
# ╠═b773915d-7bed-46eb-9277-47128a346b99
# ╟─d895b7ac-67d7-424a-b577-de74fbd9a735
# ╠═ae083570-8d89-41ee-a2a0-1a73cea63037
# ╠═a1ae3b64-e4aa-4835-a1a4-3b14a4a1c9c7
# ╠═a13aa1b2-c2e2-4ce8-aaad-6a18346c34a8
# ╠═f601491c-da58-4e7e-b7ab-9a9ca109aae7
# ╟─b154e07a-2bd4-426a-ab10-2da986304503
# ╟─488b21ad-c032-4bb6-9f13-dca382134254
# ╟─8871b0e3-0a02-4516-80c8-fb741d730eb0
# ╟─6aab0ee0-d6bf-47f6-abbf-870bab6e2385
# ╠═31e405ce-1eca-4458-9615-d142cf872e38
# ╠═fa80e92c-123b-4338-848e-e504897d64cf
# ╠═45e59e94-e174-4112-98e3-84e30701bec4
# ╟─659996e3-3f72-4cf9-89a0-004285cc321e
# ╠═21e97b17-8e7b-4f7d-8ad6-9b3f56f0c57a
# ╟─36651fa6-8bb1-4eca-8de8-93fe8d55eff0
# ╠═52dff1a0-7f79-4188-aaa6-25a563d40748
# ╟─71b13205-5459-47c7-bf1f-30b1ca2fdac3
# ╟─d4f6f713-a684-4dcb-80b0-2021b3d5902d
# ╠═67354b57-ad30-4791-942e-3ed62e9e3571
# ╠═38941b87-1e84-4345-96fa-3bc1eb9b017b
# ╠═271584dc-9617-44d2-9993-84c4aeab11c9
# ╟─47605488-5844-493c-9a69-f4e4bffa4a57
# ╠═915b2a97-377b-415b-a12a-dcd62167ac12
# ╟─c48d0ca0-f847-4fc5-9e65-7d2d82cb4e47
# ╠═a75397ea-94df-44c7-b65e-3cc2a3b499bd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
