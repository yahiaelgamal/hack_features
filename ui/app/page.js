"use client";

import React, { useState } from "react";
import ReactECharts from "echarts-for-react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ExternalLink } from "lucide-react";

import features from "./features.json";

const baseUrl =
  process.env.REACT_APP_ENV === "prod" ? "" : "http://localhost:8000";

export default function Home() {
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [featuresComparisonLink, setFeaturesComparisonLink] = useState(null);
  const [tree, setTree] = useState({});

  const options = {
    // tooltip: {
    //   trigger: "item",
    //   triggerOn: "mousemove",
    //   formatter: `
    //         Name: {b}<br/>
    //     `,
    // },
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

        label: {
          position: "left",
          verticalAlign: "middle",
          align: "right",
          fontSize: 12,
        },

        leaves: {
          label: {
            position: "right",
            verticalAlign: "middle",
            align: "left",
          },
        },

        initialTreeDepth: 10,

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
    setTree(features[value].tree);
    setFeaturesComparisonLink(getUrl(getFeatures(features[value].tree)));
  };

  return (
    <div className="flex">
      <div className="mt-5 ml-4 flex-none">
        <Select onValueChange={handleValueChange} value={selectedFeature}>
          <SelectTrigger className="w-[300px]">
            <SelectValue placeholder="Select a feature" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              <SelectLabel>Features</SelectLabel>
              {Object.keys(features).map((key) => (
                <SelectItem key={key} value={key}>
                  {`${key} (${features[key].trace_size})`}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
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
                {key}: {features[selectedFeature][key]}
              </p>
            ))}
            <div className="p-1 mt-5">
              <div>
                <a
                  className="text-blue-800 text-lg flex"
                  href={`https://www.neuronpedia.org/gpt2-small/8-res-jb/${selectedFeature
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
