import React, { useState, useEffect } from 'react';
import { DetailsList, Dropdown, Icon, IDetailsListProps, IDropdownOption, IStackTokens, Stack } from '@fluentui/react';
import ReactPaginate from 'react-paginate';

const horizontalGapStackTokens: IStackTokens = {
    childrenGap: 20,
    padding: 10
};

function _currentTableOffset(perPage: number, currentPage: number, source: any[]): number {
    return perPage === -1 ? 0 : Math.min(currentPage, Math.floor((source.length - 1) / perPage)) * perPage;
}

function _obtainPaginationSlice(perPage: number, currentPage: number, source: any[]): any[] {
    if (perPage === -1) {
        return source;
    } else {
        const offset = _currentTableOffset(perPage, currentPage, source);
        return source.slice(offset, offset + perPage);
    }
}

const PaginationTable = (props: IDetailsListProps): any => {
    const [itemsPerPage, setItemsPerPage] = useState(20 as number);
    const [currentPage, setCurrentPage] = useState(0 as number);
    // this needs to be stored in state to prevent re-rendering
    const [itemsOnPage, setItemsOnPage] = useState([] as any[]);
    const { items } = props;

    const _onItemsPerPageSelect = (event: React.FormEvent<HTMLDivElement>, item: IDropdownOption | undefined): void => {
        if (item !== undefined) {
            // use current offset to calculate the next `current_page`
            const currentOffset = _currentTableOffset(itemsPerPage, currentPage, items);
            const latestItemsPerPage = item.key as number;
            const latestCurrentPage = Math.floor(currentOffset / itemsPerPage);
            setItemsPerPage(latestItemsPerPage);
            setCurrentPage(latestCurrentPage);
            setItemsOnPage(_obtainPaginationSlice(itemsPerPage, currentPage, items));
        }
    };

    const _onPageSelect = (event: any): void => {
        setCurrentPage(event.selected);
        setItemsOnPage(_obtainPaginationSlice(itemsPerPage, event.selected, items));
    };

    useEffect(() => {
        setItemsOnPage(_obtainPaginationSlice(itemsPerPage, currentPage, items));
    }, [items, itemsPerPage]);

    const detailListProps = {
        ...props,
        items: itemsOnPage
    };
    const itemsCount = items.length;
    const pageCount = itemsPerPage === -1 ? 1 : Math.ceil(itemsCount / itemsPerPage);
    const perPageOptions = [
        { key: 10, text: '10 items per page' },
        { key: 20, text: '20 items per page' },
        { key: 50, text: '50 items per page' },
        { key: -1, text: 'All items' }
    ];
    return (
        <div>
            <DetailsList {...detailListProps} />
            <Stack
                horizontal
                horizontalAlign='end'
                verticalAlign='baseline'
                styles={{ root: { padding: 10 } }}
                tokens={horizontalGapStackTokens}
            >
                <Dropdown
                    selectedKey={itemsPerPage}
                    options={perPageOptions}
                    onChange={_onItemsPerPageSelect.bind(this)}
                    styles={{ dropdown: { width: 150 } }}
                />
                <ReactPaginate
                    previousLabel={<Icon aria-hidden={true} iconName='ChevronLeft' />}
                    nextLabel={<Icon aria-hidden={true} iconName='ChevronRight' />}
                    breakLabel={'...'}
                    breakClassName={'break'}
                    pageCount={pageCount}
                    marginPagesDisplayed={2}
                    pageRangeDisplayed={2}
                    onPageChange={_onPageSelect.bind(this)}
                    containerClassName={itemsCount === 0 ? 'pagination hidden' : 'pagination'}
                    //subContainerClassName={'pages pagination'}
                    disableInitialCallback={false}
                    activeClassName={'active'}
                />
            </Stack>
        </div>
    );
};

export default PaginationTable;
