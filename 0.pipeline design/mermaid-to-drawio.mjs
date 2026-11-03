// Convert a Mermaid flowchart into an editable draw.io diagram laid out as
// Mermaid renders it (with the repository's mermaid-config.json, as render.sh).
//
//   node "0.pipeline design/mermaid-to-drawio.mjs" INPUT.mmd [OUTPUT.drawio]
//
// Mermaid parses the file and renders it in headless Chrome; node and group
// boxes, colours and edge routes are read from the rendered SVG. Cell ids are
// the Mermaid node ids and styles record the Mermaid shape and class
// (mermaidType=..., mermaidClass=...), so edits made in draw.io can be mapped
// back onto the .mmd file. Needs `npm install` at the repository root.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const [input, output = input.replace(/\.mmd$/, ".drawio")] = process.argv.slice(2);
if (!input || !input.endsWith(".mmd")) {
  console.error("usage: mermaid-to-drawio.mjs INPUT.mmd [OUTPUT.drawio]");
  process.exit(2);
}

// draw.io shapes for Mermaid node types; "odd" (the flag) stands for a
// document in our diagrams, so it becomes draw.io's document shape.
const SHAPES = {
  square: "rounded=0;",
  round: "rounded=1;",
  stadium: "rounded=1;arcSize=50;",
  cylinder: "shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=8;",
  odd: "shape=document;boundedLbl=1;size=0.2;",
  diamond: "rhombus;",
  circle: "ellipse;",
  hexagon: "shape=hexagon;perimeter=hexagonPerimeter2;",
};
const EDGE_ENDS = {
  arrow_point: "endArrow=block;endFill=1;",
  arrow_open: "endArrow=none;",
  double_arrow_point: "startArrow=block;startFill=1;endArrow=block;endFill=1;",
};

function extract(text, config) {
  // Runs in the browser: parse, render, and measure.
  return (async () => {
    mermaid.initialize({ startOnLoad: false, ...config, flowchart: { ...config.flowchart, useMaxWidth: false } });
    const { db } = await mermaid.mermaidAPI.getDiagramFromText(text);
    const vertices = db.getVertices();
    const nodes = vertices instanceof Map ? [...vertices.values()] : Object.values(vertices);
    const { svg } = await mermaid.render("g", text);
    document.body.innerHTML = svg;
    const svgElement = document.querySelector("svg");
    const origin = svgElement.getBoundingClientRect();
    const box = (element) => {
      const r = element.getBoundingClientRect();
      return { x: r.left - origin.left, y: r.top - origin.top, width: r.width, height: r.height };
    };
    const paint = (element) => {
      if (!element) return {};
      const style = getComputedStyle(element);
      return { fill: style.fill, stroke: style.stroke, dash: style.strokeDasharray, width: style.strokeWidth };
    };
    const textColour = (element) => {
      const label = element.querySelector("text, tspan, span, div");
      return label ? getComputedStyle(label).color || getComputedStyle(label).fill : null;
    };
    // Groups that edges attach to are also listed as vertices; they are drawn as containers.
    const groupIds = new Set(db.getSubGraphs().map((group) => group.id));
    const measuredNodes = nodes.filter((node) => !groupIds.has(node.id)).map((node) => {
      const element = document.getElementById(`g-${node.domId}`);
      return {
        id: node.id, text: node.text, type: node.type, classes: node.classes,
        box: element ? box(element) : null,
        paint: element ? paint(element.querySelector("path, rect, polygon, circle, ellipse")) : {},
        textColour: element ? textColour(element) : null,
      };
    });
    const groups = db.getSubGraphs().map((group) => {
      const element = document.getElementById(`g-${group.id}`);
      return {
        id: group.id, title: group.title, members: group.nodes,
        box: element ? box(element.querySelector("rect") || element) : null,
        paint: element ? paint(element.querySelector("rect")) : {},
      };
    });
    const edges = db.getEdges().map((edge) => {
      const element = document.querySelector(`[id="g-${edge.id}"]`) || document.querySelector(`[data-id="${edge.id}"]`);
      let points = [];
      if (element && element.getTotalLength) {
        const matrix = element.getCTM();
        const total = element.getTotalLength();
        for (let length = 0; length <= total; length += Math.max(2, total / 400)) {
          const p = new DOMPoint(element.getPointAtLength(length).x, element.getPointAtLength(length).y).matrixTransform(matrix);
          points.push([p.x, p.y]);
        }
        const end = new DOMPoint(element.getPointAtLength(total).x, element.getPointAtLength(total).y).matrixTransform(matrix);
        points.push([end.x, end.y]);
      }
      return { id: edge.id, start: edge.start, end: edge.end, type: edge.type, stroke: edge.stroke, text: edge.text, points };
    });
    return { nodes: measuredNodes, groups, edges };
  })();
}

// Corners of an (almost) orthogonal sampled path: runs of horizontal and
// vertical movement, intersected pairwise.
function corners(points) {
  const runs = [];
  for (let i = 1; i < points.length; i++) {
    const dx = points[i][0] - points[i - 1][0], dy = points[i][1] - points[i - 1][1];
    if (Math.hypot(dx, dy) < 0.5) continue;
    const axis = Math.abs(dx) >= Math.abs(dy) ? "h" : "v";
    const last = runs.at(-1);
    if (last && last.axis === axis) last.points.push(points[i]);
    else runs.push({ axis, points: [points[i - 1], points[i]] });
  }
  // Ignore short runs, which are the arcs of rounded corners.
  const long = runs.filter((run) => {
    const [a, b] = [run.points[0], run.points.at(-1)];
    return Math.hypot(b[0] - a[0], b[1] - a[1]) > 6;
  });
  const merged = [];
  for (const run of long) {
    if (merged.length && merged.at(-1).axis === run.axis) merged.at(-1).points.push(...run.points);
    else merged.push({ ...run, points: [...run.points] });
  }
  const level = (run) => {
    const values = run.points.map((p) => (run.axis === "h" ? p[1] : p[0]));
    return values.reduce((s, v) => s + v, 0) / values.length;
  };
  const result = [];
  for (let i = 1; i < merged.length; i++) {
    const [a, b] = [merged[i - 1], merged[i]];
    result.push(a.axis === "v" ? [level(a), level(b)] : [level(b), level(a)]);
  }
  return result;
}

const escape = (value) => String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const colour = (css) => {
  const match = /rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/.exec(css || "");
  if (!match || match[4] === "0") return "none";
  return "#" + match.slice(1, 4).map((n) => Number(n).toString(16).padStart(2, "0")).join("");
};
const round = (n) => Math.round(n * 10) / 10;

function toDrawio(name, { nodes, groups, edges }) {
  const parentOf = {};
  for (const group of groups) for (const member of group.members) parentOf[member] = group.id;
  const boxes = Object.fromEntries([...nodes, ...groups].map((item) => [item.id, item.box]));
  const relative = (id) => {
    const b = boxes[id], parent = parentOf[id];
    const p = parent ? boxes[parent] : { x: 0, y: 0 };
    return { x: round(b.x - p.x), y: round(b.y - p.y), width: round(b.width), height: round(b.height) };
  };
  const cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>'];
  // Containers first, outermost first, so children can refer to them.
  const depth = (id) => (parentOf[id] ? 1 + depth(parentOf[id]) : 0);
  for (const group of [...groups].sort((a, b) => depth(a.id) - depth(b.id))) {
    const g = relative(group.id);
    const style = `rounded=0;whiteSpace=wrap;html=1;container=1;collapsible=0;verticalAlign=top;fontStyle=1;`
      + `fillColor=${colour(group.paint.fill)};strokeColor=${colour(group.paint.stroke)};mermaidType=subgraph;`;
    cells.push(`<mxCell id="${escape(group.id)}" value="${escape(group.title)}" style="${style}" vertex="1" parent="${escape(parentOf[group.id] || "1")}">`
      + `<mxGeometry x="${g.x}" y="${g.y}" width="${g.width}" height="${g.height}" as="geometry"/></mxCell>`);
  }
  for (const node of nodes) {
    if (!node.box) continue;
    const g = relative(node.id);
    const dashed = node.paint.dash && node.paint.dash !== "none" ? `dashed=1;dashPattern=${node.paint.dash.replace(/px|,/g, "").trim()};` : "";
    const style = (SHAPES[node.type] || SHAPES.square) + "whiteSpace=wrap;html=1;"
      + `fillColor=${colour(node.paint.fill)};strokeColor=${colour(node.paint.stroke)};${dashed}`
      + (node.textColour ? `fontColor=${colour(node.textColour)};` : "")
      + `mermaidType=${node.type};` + (node.classes.length ? `mermaidClass=${node.classes.join(" ")};` : "");
    cells.push(`<mxCell id="${escape(node.id)}" value="${escape(node.text)}" style="${style}" vertex="1" parent="${escape(parentOf[node.id] || "1")}">`
      + `<mxGeometry x="${g.x}" y="${g.y}" width="${g.width}" height="${g.height}" as="geometry"/></mxCell>`);
  }
  for (const edge of edges) {
    let style = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
      + (EDGE_ENDS[edge.type] || EDGE_ENDS.arrow_point) + (edge.stroke === "dotted" ? "dashed=1;dashPattern=2 3;" : "")
      + `mermaidType=${edge.type};`;
    let waypoints = "";
    if (edge.points.length > 1) {
      // Pin both ends where Mermaid attached them, and keep its corners.
      const anchor = (id, [px, py]) => {
        const b = boxes[id];
        const clamp = (v) => Math.min(1, Math.max(0, round(v * 100) / 100));
        return [clamp((px - b.x) / b.width), clamp((py - b.y) / b.height)];
      };
      const [ex, ey] = anchor(edge.start, edge.points[0]);
      const [nx, ny] = anchor(edge.end, edge.points.at(-1));
      style += `exitX=${ex};exitY=${ey};exitDx=0;exitDy=0;entryX=${nx};entryY=${ny};entryDx=0;entryDy=0;`;
      const points = corners(edge.points);
      if (points.length) waypoints = `<Array as="points">${points.map(([x, y]) => `<mxPoint x="${round(x)}" y="${round(y)}"/>`).join("")}</Array>`;
    }
    cells.push(`<mxCell id="${escape(edge.id)}" value="${escape(edge.text || "")}" style="${style}" edge="1" parent="1" source="${escape(edge.start)}" target="${escape(edge.end)}">`
      + `<mxGeometry relative="1" as="geometry">${waypoints}</mxGeometry></mxCell>`);
  }
  return `<mxfile host="mermaid-to-drawio">\n  <diagram name="${escape(name)}" id="${escape(name)}">\n`
    + `    <mxGraphModel grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" math="0" shadow="0">\n`
    + `      <root>\n        ${cells.join("\n        ")}\n      </root>\n    </mxGraphModel>\n  </diagram>\n</mxfile>\n`;
}

const text = fs.readFileSync(input, "utf8");
const config = JSON.parse(fs.readFileSync(path.join(root, "mermaid-config.json"), "utf8"));
const browser = await puppeteer.launch({ headless: true });
try {
  const page = await browser.newPage();
  await page.setContent("<!doctype html><html><body></body></html>");
  await page.addScriptTag({ path: path.join(root, "node_modules/mermaid/dist/mermaid.min.js") });
  const measured = await page.evaluate(extract, text, config);
  const missing = measured.nodes.filter((node) => !node.box).map((node) => node.id);
  if (missing.length) console.error(`warning: no rendered box for ${missing.join(", ")}`);
  const temporary = path.join(path.dirname(output), `.${path.basename(output)}.tmp`);
  fs.writeFileSync(temporary, toDrawio(path.basename(input, ".mmd"), measured));
  fs.renameSync(temporary, output);
  console.log(output);
} finally {
  await browser.close();
}
