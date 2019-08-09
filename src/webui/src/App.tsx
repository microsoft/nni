import * as React from 'react';
import { Row, Col } from 'antd';
import { COLUMN, COLUMNPro } from './static/const';
import './App.css';
import SlideBar from './components/SlideBar';

interface AppState {
  interval: number;
  whichPageToFresh: string;
  columnNormal: Array<string>;
  columnPro: Array<string>;
  columnList: Array<string>;
}

class App extends React.Component<{}, AppState> {
  public _isMounted: boolean;
  constructor(props: {}) {
    super(props);
    this.state = {
      interval: 10, // sendons
      whichPageToFresh: '',
      columnNormal: COLUMN,
      columnPro: COLUMNPro,
      columnList: COLUMN
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

  componentDidMount() {
    this._isMounted = true;
  }

  componentWillUnmount() {
    this._isMounted = false;
  }
  render() {
    const { interval, whichPageToFresh, columnList } = this.state;
    const reactPropsChildren = React.Children.map(this.props.children, child =>
      React.cloneElement(
        // tslint:disable-next-line:no-any
        child as React.ReactElement<any>, {
          interval, whichPageToFresh,
          columnList, changeColumn: this.changeColumn
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
