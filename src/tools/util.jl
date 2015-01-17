function mkdir_if_not_exist(dir)
    if !isdir(dir)
        println("Create $(dir)")
        run(`mkdir -p $dir`)
    end
end
