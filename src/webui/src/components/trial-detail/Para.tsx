import * as d3 from 'd3';
import { Dropdown, IDropdownOption, Stack, DefaultButton } from '@fluentui/react';
import ParCoords from 'parcoord-es';
import 'parcoord-es/dist/parcoords.css';
import * as React from 'react';
import { EXPERIMENT, TRIALS } from '../../static/datamodel';
import { SearchSpace } from '../../static/model/searchspace';
import { filterByStatus } from '../../static/function';
import { TableObj, SingleAxis, MultipleAxes } from '../../static/interface';
import '../../static/style/button.scss';
import '../../static/style/para.scss';
import ChangeColumnComponent from '../modals/ChangeColumnComponent';

interface ParaState {
    dimName: string[];
    selectedPercent: string;
    primaryMetricKey: string;
    noChart: boolean;
    customizeColumnsDialogVisible: boolean;
    availableDimensions: string[];
    chosenDimensions: string[];
}

interface ParaProps {
    trials: Array<TableObj>;
    searchSpace: SearchSpace;
    whichChart: string;
}

class Para extends React.Component<ParaProps, ParaState> {
    private paraRef = React.createRef<HTMLDivElement>();
    private pcs: any;

    private chartMulineStyle = {
        width: '100%',
        height: 392,
        margin: '0 auto'
    };
    private innerChartMargins = {
        top: 32,
        right: 20,
        bottom: 20,
        left: 28
    };

    constructor(props: ParaProps) {
        super(props);
        this.state = {
            dimName: [],
            primaryMetricKey: 'default',
            selectedPercent: '1',
            noChart: true,
            customizeColumnsDialogVisible: false,
            availableDimensions: [],
            chosenDimensions: []
        };
    }

    // get percent value number
    percentNum = (event: React.FormEvent<HTMLDivElement>, item?: IDropdownOption): void => {
        if (item !== undefined) {
            this.setState({ selectedPercent: item.key.toString() }, () => {
                this.renderParallelCoordinates();
            });
        }
    };

    // select all final keys
    updateEntries = (event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            this.setState({ primaryMetricKey: item.key }, () => {
                this.renderParallelCoordinates();
            });
        }
    };

    componentDidMount(): void {
        this.renderParallelCoordinates();
    }

    componentDidUpdate(prevProps: ParaProps): void {
        // FIXME: redundant update
        if (this.props.trials !== prevProps.trials || this.props.searchSpace !== prevProps.searchSpace) {
            const { whichChart } = this.props;
            if (whichChart === 'Hyper-parameter') {
                this.renderParallelCoordinates();
            }
        }
    }

    render(): React.ReactNode {
        const {
            selectedPercent,
            noChart,
            customizeColumnsDialogVisible,
            availableDimensions,
            chosenDimensions
        } = this.state;

        return (
            <div className='parameter'>
                <Stack horizontal className='para-filter' horizontalAlign='end'>
                    <DefaultButton
                        text='Add/Remove axes'
                        onClick={(): void => {
                            this.setState({ customizeColumnsDialogVisible: true });
                        }}
                        styles={{ root: { marginRight: 10 } }}
                    />
                    <Dropdown
                        selectedKey={selectedPercent}
                        onChange={this.percentNum}
                        options={[
                            { key: '0.01', text: 'Top 1%' },
                            { key: '0.05', text: 'Top 5%' },
                            { key: '0.2', text: 'Top 20%' },
                            { key: '1', text: 'Top 100%' }
                        ]}
                        styles={{ dropdown: { width: 120 } }}
                        className='para-filter-percent'
                    />
                    {this.finalKeysDropdown()}
                </Stack>
                {customizeColumnsDialogVisible && availableDimensions.length > 0 && (
                    <ChangeColumnComponent
                        selectedColumns={chosenDimensions}
                        allColumns={availableDimensions.map(dim => ({ key: dim, name: dim }))}
                        onSelectedChange={(selected: string[]): void => {
                            this.setState({ chosenDimensions: selected }, () => {
                                this.renderParallelCoordinates();
                            });
                        }}
                        onHideDialog={(): void => {
                            this.setState({ customizeColumnsDialogVisible: false });
                        }}
                        minSelected={2}
                    />
                )}
                <div className='parcoords' style={this.chartMulineStyle} ref={this.paraRef} />
                {noChart && <div className='nodata'>No data</div>}
            </div>
        );
    }

    private finalKeysDropdown(): any {
        const { primaryMetricKey } = this.state;
        if (TRIALS.finalKeys().length === 1) {
            return null;
        } else {
            const finalKeysDropdown: any = [];
            TRIALS.finalKeys().forEach(item => {
                finalKeysDropdown.push({
                    key: item,
                    text: item
                });
            });
            return (
                <div>
                    <span className='para-filter-text para-filter-middle'>Metrics</span>
                    <Dropdown
                        selectedKey={primaryMetricKey}
                        options={finalKeysDropdown}
                        onChange={this.updateEntries}
                        styles={{ root: { width: 150, display: 'inline-block' } }}
                        className='para-filter-percent'
                    />
                </div>
            );
        }
    }

    /**
     * Render the parallel coordinates. Using trial data as base and leverage
     * information from search space at a best effort basis.
     * @param source Array of trial data
     * @param searchSpace Search space
     */
    private renderParallelCoordinates(): void {
        const { searchSpace } = this.props;
        const percent = parseFloat(this.state.selectedPercent);
        const { primaryMetricKey, chosenDimensions } = this.state;

        const inferredSearchSpace = TRIALS.inferredSearchSpace(searchSpace);
        const inferredMetricSpace = TRIALS.inferredMetricSpace();
        let convertedTrials = this.getTrialsAsObjectList(inferredSearchSpace, inferredMetricSpace);

        const dimensions: [string, any][] = [];
        let colorDim: string | undefined = undefined,
            colorScale: any = undefined;
        // treat every axis as numeric to fit for brush
        for (const [k, v] of inferredSearchSpace.axes) {
            dimensions.push([
                k,
                {
                    type: 'number',
                    yscale: this.convertToD3Scale(v)
                }
            ]);
        }
        for (const [k, v] of inferredMetricSpace.axes) {
            const scale = this.convertToD3Scale(v);
            if (k === primaryMetricKey && scale !== undefined && scale.interpolate) {
                // set color for primary metrics
                // `colorScale` is used to produce a color range, while `scale` is to produce a pixel range
                colorScale = this.convertToD3Scale(v, false);
                convertedTrials.sort((a, b) => (EXPERIMENT.optimizeMode === 'minimize' ? a[k] - b[k] : b[k] - a[k]));
                // filter top trials
                if (percent != 1) {
                    const keptTrialNum = Math.max(Math.ceil(convertedTrials.length * percent), 1);
                    convertedTrials = convertedTrials.slice(0, keptTrialNum);
                    const domain = d3.extent(convertedTrials, item => item[k]);
                    scale.domain([domain[0], domain[1]]);
                    colorScale.domain([domain[0], domain[1]]);
                    if (colorScale !== undefined) {
                        colorScale.domain(domain);
                    }
                }
                // reverse the converted trials to show the top ones upfront
                convertedTrials.reverse();
                const assignColors = (scale: any): void => {
                    scale.range([0, 1]); // fake a range to perform invert
                    const [scaleMin, scaleMax] = scale.domain();
                    const pivot = scale.invert(0.5);
                    scale
                        .domain([scaleMin, pivot, scaleMax])
                        .range(['#90EE90', '#FFC400', '#CA0000'])
                        .interpolate(d3.interpolateHsl);
                };
                assignColors(colorScale);
                colorDim = k;
            }
            dimensions.push([
                k,
                {
                    type: 'number',
                    yscale: scale
                }
            ]);
        }

        if (convertedTrials.length === 0 || dimensions.length <= 1) {
            return;
        }

        const firstRun = this.pcs === undefined;
        if (firstRun) {
            this.pcs = ParCoords()(this.paraRef.current);
        }
        this.pcs
            .data(convertedTrials)
            .dimensions(
                dimensions
                    .filter(([d, _]) => chosenDimensions.length === 0 || chosenDimensions.includes(d))
                    .reduce((obj, entry) => ({ ...obj, [entry[0]]: entry[1] }), {})
            );
        if (firstRun) {
            this.pcs
                .margin(this.innerChartMargins)
                .alphaOnBrushed(0.2)
                .smoothness(0.1)
                .brushMode('1D-axes')
                .reorderable()
                .interactive();
        }
        if (colorScale !== undefined) {
            this.pcs.color(d => (colorScale as any)(d[colorDim as any]));
        }
        this.pcs.render();
        if (firstRun) {
            this.setState({ noChart: false });
        }

        // set new available dims
        this.setState({
            availableDimensions: dimensions.map(e => e[0]),
            chosenDimensions: chosenDimensions.length === 0 ? dimensions.map(e => e[0]) : chosenDimensions
        });
    }

    private getTrialsAsObjectList(inferredSearchSpace: MultipleAxes, inferredMetricSpace: MultipleAxes): {}[] {
        const { trials } = this.props;
        const succeededTrials = trials.filter(filterByStatus);

        return succeededTrials.map(s => {
            const entries = Array.from(s.parameters(inferredSearchSpace).entries());
            entries.push(...Array.from(s.metrics(inferredMetricSpace).entries()));
            const ret = {};
            for (const [k, v] of entries) {
                ret[k.fullName] = v;
            }
            return ret;
        });
    }

    private getRange(): [number, number] {
        // Documentation is lacking.
        // Reference: https://github.com/syntagmatic/parallel-coordinates/issues/308
        // const range = this.pcs.height() - this.pcs.margin().top - this.pcs.margin().bottom;
        const range = this.chartMulineStyle.height - this.innerChartMargins.top - this.innerChartMargins.bottom;
        return [range, 1];
    }

    private convertToD3Scale(axis: SingleAxis, initRange: boolean = true): any {
        const padLinear = ([x0, x1], k = 0.1): [number, number] => {
            const dx = ((x1 - x0) * k) / 2;
            return [x0 - dx, x1 + dx];
        };
        const padLog = ([x0, x1], k = 0.1): [number, number] => {
            const [y0, y1] = padLinear([Math.log(x0), Math.log(x1)], k);
            return [Math.exp(y0), Math.exp(y1)];
        };
        let scaleInst: any = undefined;
        if (axis.scale === 'ordinal') {
            if (axis.nested) {
                // TODO: handle nested entries
                scaleInst = d3
                    .scalePoint()
                    .domain(Array.from(axis.domain.keys()))
                    .padding(0.2);
            } else {
                scaleInst = d3
                    .scalePoint()
                    .domain(axis.domain)
                    .padding(0.2);
            }
        } else if (axis.scale === 'log') {
            scaleInst = d3.scaleLog().domain(padLog(axis.domain));
        } else if (axis.scale === 'linear') {
            scaleInst = d3.scaleLinear().domain(padLinear(axis.domain));
        }
        if (initRange) {
            scaleInst = scaleInst.range(this.getRange());
        }
        return scaleInst;
    }
}

export default Para;
