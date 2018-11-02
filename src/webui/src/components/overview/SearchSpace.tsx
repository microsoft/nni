import * as React from 'react';

interface SearchspaceProps {
    searchSpace: object;
}

class SearchSpace extends React.Component<SearchspaceProps, {}> {

    constructor(props: SearchspaceProps) {
        super(props);

    }

    render() {
        const { searchSpace } = this.props;
        return (
            <pre className="experiment searchSpace" style={{paddingLeft: 20}}>
                {JSON.stringify(searchSpace, null, 4)}
            </pre>
        );
    }
}

export default SearchSpace;