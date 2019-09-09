import * as React from 'react';
import { Row, Col } from 'antd';
import axios from 'axios';
import { COLUMN, MANAGER_IP } from './static/const';
import './App.css';
import SlideBar from './components/SlideBar';

interface AppState {
  interval: number;
  whichPageToFresh: string;
  columnList: Array<string>;
  concurrency: number;
}

class App extends React.Component<{}, AppState> {
  public _isMounted: boolean;
  constructor(props: {}) {
    super(props);
    this.state = {
      interval: 10, // sendons
      whichPageToFresh: '',
      columnList: COLUMN,
      concurrency: 1
    };
  }

  changeInterval = (interval: number) => {
    if (this._isMounted === true) {
      this.setState(() => ({ interval: interval }));
    }
  }

  changeFresh = (fresh: string) => {
    // interval * 1000 
    if (this._isMounted === true) {
      this.setState(() => ({ whichPageToFresh: fresh }));
    }
  }

  changeColumn = (columnList: Array<string>) => {
    if (this._isMounted === true) {
      this.setState(() => ({ columnList: columnList }));
    }
  }

  changeConcurrency = (val: number) => {
    if (this._isMounted === true) {
      this.setState(() => ({ concurrency: val }));
    }
  }

  getConcurrency = () => {
    axios(`${MANAGER_IP}/experiment`, {
      method: 'GET'
    })
      .then(res => {
        if (res.status === 200) {
          const params = res.data.params;
          if (this._isMounted) {
            this.setState(() => ({ concurrency: params.trialConcurrency }));
          }
        }
      });
  }

  componentDidMount() {
    this._isMounted = true;
    this.getConcurrency();
  }

  componentWillUnmount() {
    this._isMounted = false;
  }
  render() {
    const { interval, whichPageToFresh, columnList, concurrency } = this.state;
    const reactPropsChildren = React.Children.map(this.props.children, child =>
      React.cloneElement(
        // tslint:disable-next-line:no-any
        child as React.ReactElement<any>, {
          interval, whichPageToFresh,
          columnList, changeColumn: this.changeColumn,
          concurrency, changeConcurrency: this.changeConcurrency
        })
    );
    return (
      <Row className="nni" style={{ minHeight: window.innerHeight }}>
        <Row className="header">
          <Col span={1} />
          <Col className="headerCon" span={22}>
            <SlideBar changeInterval={this.changeInterval} changeFresh={this.changeFresh} />
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
}

export default App;
