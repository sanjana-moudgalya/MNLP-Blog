import { searchPlugin } from "@vuepress/plugin-search";
import { defineUserConfig } from "vuepress";
import theme from "./theme.js";

export default defineUserConfig({
  lang: "en-US",
  title: "Evaluation of Machine Translations with Large Language Models",
  description: "A Blog for Machine Learning, Natural Language Processing, and Data Mining",
  base: "/blog/",
  theme,
  plugins: [
    searchPlugin({
      // your options
    }),
  ],
});
