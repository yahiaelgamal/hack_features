"use client";

import React, { useState } from "react";
import ReactECharts from "echarts-for-react";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { CaretSortIcon, CheckIcon } from "@radix-ui/react-icons";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ExternalLink } from "lucide-react";

import features from "./features_annotated.json";

const featuresData = Object.keys(features).map((f) => ({
  value: f,
  label: `${f} (${features[f].trace_size})`,
}));

export default function Home() {
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [selectedFeatureData, setSelectedFeatureData] = useState(null);
  const [featuresComparisonLink, setFeaturesComparisonLink] = useState(null);
  const [open, setOpen] = useState(false);
  const [tree, setTree] = useState({});

  const options = {
    series: [
      {
        type: "tree",
        name: "features",

        data: [tree],

        top: "15%",
        left: "10%",
        bottom: "10%",
        right: "50%",

        // height: "100%",
        // width: "600px",

        zoom: 1,

        symbolSize: 8,
        symbol: "circle",

        layout: "orthogonal",

        orient: "LR", // vertical
        edgeShape: "polyline", // "curve",
        edgeSymbol: ["", "arrow"],

        label: {
          position: "left",
          verticalAlign: "bottom",
          align: "right",
          fontSize: 12,
          offset: [0, -5],
          formatter: function (params) {
            const desc = params.data.desc;
            const name = params.data.name;
            if (desc != null) {
              return `{name|${name}}\n{desc|${desc}}`;
            }
            return name;
          },
          rich: {
            name: {
              fontSize: 12,
              fontWeight: "bold",
              color: "#333",
              align: "right",
              padding: [0, 0, 5, 0],
            },
            desc: {
              fontSize: 12,
              color: "#333",
              align: "right",
            },
          },
        },

        leaves: {
          label: {
            position: "right",
            verticalAlign: "middle",
            align: "left",
          },
        },

        initialTreeDepth: 5,

        emphasis: {
          focus: "descendant", // ancestor
          // itemStyle: {
          //   opacity: 0.9, // Set opacity to 1 for hovered nodes
          // },
        },

        roam: true,

        expandAndCollapse: true,

        animationDuration: 200,
        animationDurationUpdate: 300,
      },
    ],
  };

  function getFeatures(tree) {
    const features = [];
    const threshold = 0;

    function traverse(node) {
      const [layer, feature] = node.name.split("/");
      if (
        node.value >= threshold ||
        node.children.filter((n) => n.value >= threshold).length > 0
      ) {
        features.push({
          layer: layer.substring(1),
          feature: feature.substring(1),
        });
      }

      if (node.children) {
        node.children.forEach(traverse);
      }
    }

    if (tree.name) {
      traverse(tree);
    }
    return features;
  }

  function getUrl(features) {
    let url =
      "https://neuronpedia.org/quick-list/?name=feature-comparison-temp";
    const formattedFeatures = features.map((feature) => ({
      modelId: "gpt2-small",
      layer: `${feature.layer}-res-jb`,
      index: feature.feature,
    }));
    url += "&features=" + encodeURIComponent(JSON.stringify(formattedFeatures));
    return url;
  }

  const handleValueChange = (value) => {
    setSelectedFeature(value);
    setSelectedFeatureData({ ...features[value], tree: null });
    setTree(features[value].tree);
    setFeaturesComparisonLink(getUrl(getFeatures(features[value].tree)));
  };

  return (
    <div className="flex">
      <div className="mt-5 ml-4 flex-none">
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <Button
              variant="outline"
              role="combobox"
              aria-expanded={open}
              className="w-[300px] justify-between"
            >
              {selectedFeature
                ? featuresData.find(
                    (feature) => feature.value === selectedFeature
                  )?.label
                : "Select Feature"}
              <CaretSortIcon className="ml-2 h-4 w-4 shrink-0 opacity-50" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[300px] p-2">
            <Command>
              <CommandInput placeholder="Search Feature..." className="h-9" />
              <CommandEmpty>No features found.</CommandEmpty>
              <CommandGroup className="overflow-auto max-h-[400px]">
                {featuresData.map((feature) => (
                  <CommandItem
                    key={feature.value}
                    value={feature.value}
                    onSelect={(currentValue) => {
                      handleValueChange(
                        currentValue === selectedFeature
                          ? ""
                          : currentValue.toUpperCase()
                      );
                      setOpen(false);
                    }}
                  >
                    {feature.label}
                    <CheckIcon
                      className={cn(
                        "ml-auto h-4 w-4",
                        selectedFeature === feature.value
                          ? "opacity-100"
                          : "opacity-0"
                      )}
                    />
                  </CommandItem>
                ))}
              </CommandGroup>
            </Command>
          </PopoverContent>
        </Popover>

        {selectedFeature && (
          <div className="mt-5 ">
            {[
              "feature_id",
              "trace_size",
              "degree",
              "degree_rank",
              "degree_centrality_rank",
              // "betweenness_centrality_rank",
              "pagerank_rank",
            ].map((key) => (
              <p
                key={key}
                className="p-1 bg-slate-50 text-slate-900 text-md font-condensed"
              >
                {key}: {selectedFeatureData[key]}
              </p>
            ))}
            <div className="p-1 mt-5">
              <div>
                <a
                  className="text-blue-800 text-lg flex"
                  href={`https://www.neuronpedia.org/gpt2-small/${selectedFeature
                    .split("/")[0]
                    .slice(1)}-res-jb/${selectedFeature
                    .split("/")[1]
                    .slice(1)}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Neuronpedia Feature Page <ExternalLink className="ml-2" />
                </a>
              </div>
              <div className="mt-2">
                <a
                  className="text-blue-800 text-lg flex "
                  href={featuresComparisonLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Neuronpedia Feature Comparison{" "}
                  <ExternalLink className="ml-2" />
                </a>
              </div>
            </div>
          </div>
        )}
      </div>
      <div className="flex-none w-full">
        {selectedFeature && (
          <div>
            <div id="treeWrapper" style={{ width: "100%" }}>
              <ReactECharts
                style={{ height: "100vh", width: "100%" }}
                option={options}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
