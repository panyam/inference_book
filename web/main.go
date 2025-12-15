package main

import (
	"flag"
	"log"
	"net/http"
	"os"

	"github.com/felixge/httpsnoop"
	"github.com/gorilla/mux"
	s3 "github.com/panyam/s3gen"
)

var (
	addr = flag.String("addr", DefaultAddress(), "Address to serve on")
)

var site = s3.Site{
	OutputDir:   "./output",
	ContentRoot: "./content",
	TemplateFolders: []string{
		"./templates",
	},
	StaticFolders: []string{
		"/static/", "static",
	},
	DefaultBaseTemplate: s3.BaseTemplate{
		Name: "CalculatorBase.html",
	},
	AssetPatterns: []string{
		"*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.webp",
		"*.js", "*.css", "*.json",
	},
	BuildRules: []s3.Rule{},
}

func main() {
	flag.Parse()

	if os.Getenv("APP_ENV") != "production" {
		site.Watch()
	}

	router := mux.NewRouter()
	router.PathPrefix(site.PathPrefix).Handler(http.StripPrefix(site.PathPrefix, &site))

	srv := &http.Server{
		Handler: withLogger(router),
		Addr:    *addr,
	}
	log.Printf("'Inference is all you need' Book Calculators serving on %s", *addr)
	log.Fatal(srv.ListenAndServe())
}

func DefaultAddress() string {
	port := os.Getenv("CALC_PORT")
	if port != "" {
		return port
	}
	return ":8088"
}

func withLogger(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		m := httpsnoop.CaptureMetrics(handler, writer, request)
		log.Printf("http[%d] %s %s\n", m.Code, m.Duration, request.URL.Path)
	})
}
