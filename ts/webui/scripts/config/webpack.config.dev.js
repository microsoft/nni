'use strict';

const resolve = require('resolve');

const HtmlWebpackPlugin = require('html-webpack-plugin');
const MonacoWebpackPlugin = require('monaco-editor-webpack-plugin');
const InterpolateHtmlPlugin = require('react-dev-utils/InterpolateHtmlPlugin');
const ModuleNotFoundPlugin = require('react-dev-utils/ModuleNotFoundPlugin');

const getClientEnvironment = require('./env');
const paths = require('./paths');

const publicPath = './';
const publicUrl = '.';
const env = getClientEnvironment(publicUrl);

const config = {
    mode: 'development',
    devtool: 'cheap-module-source-map',
    entry: [ paths.appIndexJs ],
    output: {
        path: paths.appBuild,
        pathinfo: true,
        filename: 'static/js/bundle.js',
        chunkFilename: 'static/js/[name].chunk.js',
        publicPath: publicPath,
    },
    optimization: { minimize: false },
    resolve: {
        modules: [ 'node_modules' ],
        extensions: paths.moduleFileExtensions.map(ext => `.${ext}`),
    },
    module: {
        strictExportPresence: true,
        rules: [
            {
                oneOf: [
                    {
                        test: [ /\.bmp$/, /\.gif$/, /\.jpe?g$/, /\.png$/ ],
                        loader: require.resolve('url-loader'),
                        options: { limit: 10000, name: 'static/media/[name].[hash:8].[ext]' },
                    },
                    {
                        test: /\.(js|mjs|jsx|ts|tsx)$/,
                        include: paths.appSrc,
                        loader: require.resolve('babel-loader'),
                        options: { cacheDirectory: true, cacheCompression: false, compact: false },
                    },
                    {
                        test: /\.css$/,
                        use: [
                            require.resolve('style-loader'),
                            require.resolve('css-loader'),
                        ],
                        sideEffects: true,
                    },
                    {
                        test: /\.(scss|sass)$/,
                        use: [
                            require.resolve('style-loader'),
                            require.resolve('css-loader'),
                            require.resolve('sass-loader'),
                        ],
                        sideEffects: true,
                    },
                    {
                        loader: require.resolve('file-loader'),
                        exclude: [/\.(js|mjs|jsx|ts|tsx)$/, /\.html$/, /\.json$/],
                        options: { name: 'static/media/[name].[hash:8].[ext]' },
                    },
                ],
            },
        ],
    },
    plugins: [
        new HtmlWebpackPlugin({ inject: true, template: paths.appHtml }),
        new MonacoWebpackPlugin({ languages: [ 'json' ] }),
        new InterpolateHtmlPlugin(HtmlWebpackPlugin, env.raw),
        new ModuleNotFoundPlugin(paths.appPath),
    ],
    performance: { hints: false },
}

module.exports = () => config;
