import * as d3 from 'd3';
import { Dropdown, IDropdownOption, PrimaryButton, Stack } from 'office-ui-fabric-react';
import ParCoords from 'parcoord-es';
import 'parcoord-es/dist/parcoords.css';
import * as React from 'react';
import { TRIALS, EXPERIMENT } from '../../static/datamodel';
import { SearchSpace } from '../../static/model/searchspace';
import { filterByStatus } from '../../static/function';
import { ParaObj, TableObj, SingleAxis } from '../../static/interface';
import '../../static/style/button.scss';
import '../../static/style/para.scss';

function isSearchSpaceNested(searchSpace): boolean {
    for (const item of Object.values(searchSpace)) {
        const value = (item as any)._value;
        if (value && typeof value[0] === 'object') {
            return true;
        }
    }
    return false;
}

interface ParaState {
    // paraSource: Array<TableObj>;
    option: object;
    paraBack: ParaObj;
    dimName: string[];
    swapAxisArr: string[];
    percent: number;
    paraNodata: string;
    max: number; // graph color bar limit
    min: number;
    sutrialCount: number; // succeed trial numbers for SUC
    succeedRenderCount: number; // all succeed trials number
    clickCounts: number;
    isLoadConfirm: boolean;
    // office-fabric-ui
    selectedItem?: { key: string | number | undefined }; // percent Selector
    swapyAxis?: string[]; // yAxis Selector
    paraYdataNested: number[][];
    isNested: boolean;
    showFinalMetricKey: string;
    metricType: string;
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
        top: 24,
        right: 12,
        bottom: 20,
        left: 12
    };

    constructor(props: ParaProps) {
        super(props);
        this.state = {
            // paraSource: [],
            // option: this.hyperParaPic,
            option: {},
            dimName: [],
            paraBack: {
                parallelAxis: [{
                    dim: 0,
                    name: ''
                }],
                data: []
            },
            swapAxisArr: [],
            percent: 0,
            paraNodata: '',
            min: 0,
            max: 1,
            sutrialCount: 10000000,
            succeedRenderCount: 10000000,
            clickCounts: 1,
            isLoadConfirm: false,
            swapyAxis: [],
            paraYdataNested: [],
            isNested: false,
            showFinalMetricKey: "default",
            metricType: 'numberType'
        };
    }

    // get percent value number
    // percentNum = (value: string) => {
    percentNum = (event: React.FormEvent<HTMLDivElement>, item?: IDropdownOption): void => {
        // percentNum = (event: React.FormEvent<HTMLDivElement>, item?: ISelectableOption) => {
        if (item !== undefined) {
            const vals = parseFloat(item !== undefined ? item.text : '');
            this.setState({ percent: vals / 100, selectedItem: item }, () => {
                this.renderParallelCoordinates();
            });
        }
    }

    // select all final keys
    updateEntries = (event: React.FormEvent<HTMLDivElement>, item: any): void => {
        if (item !== undefined) {
            this.setState({ showFinalMetricKey: item.key }, () => {
                this.renderParallelCoordinates();
            });
        }
    }

    componentDidMount(): void {
        this.renderParallelCoordinates();
    }

    componentDidUpdate(prevProps: ParaProps): void {
        if (this.props.trials !== prevProps.trials) {
            const { whichChart } = this.props;
            if (whichChart === 'Hyper-parameter') {
                this.renderParallelCoordinates();
            }
        }
    }

    render(): React.ReactNode {
        const { option, paraNodata, dimName, isLoadConfirm, selectedItem, swapyAxis } = this.state;

        return (
            <div className="parameter">
                <Stack horizontal className="para-filter" horizontalAlign="end">
                    <span className="para-filter-text">Top</span>
                    <Dropdown
                        selectedKey={selectedItem ? selectedItem.key : undefined}
                        onChange={this.percentNum}
                        placeholder="100%"
                        defaultSelectedKeys={[0.2]}
                        options={[
                            { key: '0.2', text: '20%' },
                            { key: '0.5', text: '50%' },
                            { key: '0.8', text: '80%' },
                            { key: '1', text: '100%' },
                        ]}
                        styles={{ dropdown: { width: 120 } }}
                        className="para-filter-percent"
                    />
                    {this.finalKeysDropdown()}
                    <PrimaryButton
                        text="Confirm"
                        disabled={isLoadConfirm}
                    />
                </Stack>
                <div className="parcoords" style={this.chartMulineStyle} ref={this.paraRef} />
            </div>
        );
    }

    private finalKeysDropdown = (): any => {
        const { showFinalMetricKey } = this.state;
        if (TRIALS.finalKeys().length === 1) {
            return null;
        } else {
            const finalKeysDropdown: any = [];
            TRIALS.finalKeys().forEach(item => {
                finalKeysDropdown.push({
                    key: item, text: item
                });
            });
            return (
                <div>
                    <span className="para-filter-text para-filter-middle">Metrics</span>
                    <Dropdown
                        selectedKey={showFinalMetricKey}
                        options={finalKeysDropdown}
                        onChange={this.updateEntries}
                        styles={{ root: { width: 150, display: 'inline-block' } }}
                        className="para-filter-percent"
                    />
                </div>
            );
        }
    };

    /**
     * Render the parallel coordinates. Using trial data as base and leverage
     * information from search space at a best effort basis.
     * @param source Array of trial data
     * @param searchSpace Search space
     */
    private renderParallelCoordinates = (): void => {
        const { trials, searchSpace } = this.props;
        const convertToD3Scale = (axis: SingleAxis) => {
            if (axis.scale === 'ordinal') {
                return d3.scalePoint().domain(axis.domain).range(this.getRange());
            } else if (axis.scale === 'log') {
                return d3.scaleLog().domain(axis.domain).range(this.getRange());
            } else if (axis.scale === 'linear') {
                return d3.scaleLinear().domain(axis.domain).range(this.getRange());
            }
        };
        // filter succeed trials [{}, {}, {}]
        const succeededTrials = trials.filter(filterByStatus);
        const convertedTrials = succeededTrials.map(s => {
            const entries = Array.from(s.parameters(searchSpace.getAxesTree()).entries());
            entries.push(...(Array.from(s.metrics().entries())));
            const ret = {};
            for (const [k, v] of entries) {
                ret[k.fullName] = v;
            }
            return ret;
        });
        const inferredSearchSpace = TRIALS.inferredSearchSpace(searchSpace);
        const inferredMetricSpace = TRIALS.inferredMetricSpace();
        const dimensions: [any, any][] = [];
        // treat all as number to fit for brush
        for (const [k, v] of inferredSearchSpace.getAllAxes()) {
            dimensions.push([k.fullName, {
                type: 'number',
                yscale: convertToD3Scale(v)
            }]);
        }
        for (const [k, v] of inferredMetricSpace.getAllAxes()) {
            // const title = `metrics/${k}`;
            dimensions.push([k.fullName, {
                type: 'number',
                yscale: convertToD3Scale(v)
            }]);
        }

        if (this.pcs === undefined) {
            this.pcs = ParCoords()(this.paraRef.current)
                .data(convertedTrials)
                .showControlPoints(false)
                .margin(this.innerChartMargins)
                .dimensions(dimensions.reduce((obj, entry) => ({...obj, [entry[0]]: entry[1]}), {}))
                .alphaOnBrushed(0.2)
                .render()
                .brushMode("1D-axes")
                .reorderable()
                .interactive();
        }
    }

    private getRange(): [number, number] {
        // Documentation is lacking.
        // Reference: https://github.com/syntagmatic/parallel-coordinates/issues/308
        // const range = this.pcs.height() - this.pcs.margin().top - this.pcs.margin().bottom;
        const range = this.chartMulineStyle.height - this.innerChartMargins.top - this.innerChartMargins.bottom;
        return [range, 1];
    }

}

export default Para;
