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
import { Badge } from "@/components/ui/badge";

import features from "./features.json";

const baseUrl =
  process.env.REACT_APP_ENV === "prod" ? "" : "http://localhost:8000";

export default function Home() {
  const [selectedFeature, setSelectedFeature] = useState(null);
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

        orient: "LR", // Set the orientation to left-to-right

        label: {
          position: "left",
          verticalAlign: "middle",
          align: "right",
          textStyle: { fontSize: 12 },
        },

        leaves: {
          label: {
            position: "right",
            verticalAlign: "middle",
            align: "left",
          },
        },

        initialTreeDepth: 12,

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

  const handleValueChange = (value) => {
    setSelectedFeature(value);
    setTree(features[value].tree);
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
          <div className="mt-5">
            {[
              "feature_id",
              "trace_size",
              "degree",
              "degree_rank",
              "degree_centrality_rank",
              "betweenness_centrality_rank",
              "pagerank",
            ].map((key) => (
              <p className="p-1 bg-slate-50 text-slate-900">
                <b>{key}</b>: {features[selectedFeature][key]}
              </p>
            ))}
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
