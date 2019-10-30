import * as React from 'react';
import { Row, Col } from 'antd';
import { COLUMN } from './static/const';
import { EXPERIMENT, TRIALS } from './static/datamodel';
import './App.css';
import SlideBar from './components/SlideBar';

interface AppState {
    interval: number;
    columnList: Array<string>;
    experimentUpdateBroadcast: number;
    trialsUpdateBroadcast: number;
}

class App extends React.Component<{}, AppState> {
    private timerId: number | null;

    constructor(props: {}) {
        super(props);
        this.state = {
            interval: 10, // sendons
            columnList: COLUMN,
            experimentUpdateBroadcast: 0,
            trialsUpdateBroadcast: 0,
        };
    }

    async componentDidMount() {
        await Promise.all([ EXPERIMENT.init(), TRIALS.init() ]);
        this.setState(state => ({ experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1 }));
        this.setState(state => ({ trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1 }));
        this.timerId = window.setTimeout(this.refresh, this.state.interval * 1000);
    }

    changeInterval = (interval: number) => {
        this.setState({ interval });
        if (this.timerId === null && interval !== 0) {
            window.setTimeout(this.refresh);
        } else if (this.timerId !== null && interval === 0) {
            window.clearTimeout(this.timerId);
        }
    }

    // TODO: use local storage
    changeColumn = (columnList: Array<string>) => {
        this.setState({ columnList: columnList });
    }

    render() {
        const { interval, columnList, experimentUpdateBroadcast, trialsUpdateBroadcast } = this.state;
        if (experimentUpdateBroadcast === 0 || trialsUpdateBroadcast === 0) {
            return null;  // TODO: render a loading page
        }
        const reactPropsChildren = React.Children.map(this.props.children, child =>
            React.cloneElement(
                // tslint:disable-next-line:no-any
                child as React.ReactElement<any>, {
                    interval,
                    columnList, changeColumn: this.changeColumn,
                    experimentUpdateBroadcast,
                    trialsUpdateBroadcast,
                })
        );
        return (
            <Row className="nni" style={{ minHeight: window.innerHeight }}>
                <Row className="header">
                    <Col span={1} />
                    <Col className="headerCon" span={22}>
                        <SlideBar changeInterval={this.changeInterval} />
                    </Col>
                    <Col span={1} />
                </Row>
                <Row className="contentBox">
                    <Row className="content">
                        {reactPropsChildren}
                    </Row>
                </Row>
            </Row>
        );
    }

    private refresh = async () => {
        const [ experimentUpdated, trialsUpdated ] = await Promise.all([ EXPERIMENT.update(), TRIALS.update() ]);
        if (experimentUpdated) {
            this.setState(state => ({ experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1 }));
        }
        if (trialsUpdated) {
            this.setState(state => ({ trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1 }));
        }

        if ([ 'DONE', 'ERROR', 'STOPPED' ].includes(EXPERIMENT.status)) {
            // experiment finished, refresh once more to ensure consistency
            if (this.state.interval > 0) {
                this.setState({ interval: 0 });
                this.lastRefresh();
            }

        } else if (this.state.interval !== 0) {
            this.timerId = window.setTimeout(this.refresh, this.state.interval * 1000);
        }
    }

    private async lastRefresh() {
        await EXPERIMENT.update();
        await TRIALS.update(true);
        this.setState(state => ({ experimentUpdateBroadcast: state.experimentUpdateBroadcast + 1 }));
        this.setState(state => ({ trialsUpdateBroadcast: state.trialsUpdateBroadcast + 1 }));
    }
}

export default App;
